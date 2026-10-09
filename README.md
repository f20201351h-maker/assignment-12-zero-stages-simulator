# ZeRO-0 to ZeRO-3 on 32 virtual GPUs

Built as ERA V5 Session 12 assignment. "The notes" below are that session's course notes, which supply the 16-byte table and the 30B memory ladder I compare against.

I built 32 virtual GPUs on my laptop's CPU, trained a small GPT on them under ordinary data parallelism (ZeRO-0) and ZeRO stages 1, 2 and 3, and checked three things: what each rank actually stores, what has to be sent between ranks, and whether the four setups produce the same weight update. Everything runs on CPU in a minute or two. The executed notebook is [`ZeRO_32_virtual_GPUs.ipynb`](ZeRO_32_virtual_GPUs.ipynb).

The short version, measured on rank 11 of 32:

| | rank 11 keeps between steps | bytes / param | Adam runs on | sent per step | update vs DP |
|---|---|---|---|---|---|
| DP (ZeRO-0) | 13.23 MiB | 16 | 867,072 params | 1.94 P | reference |
| ZeRO-1 | 3.62 MiB | 4.375 | 27,096 | 1.94 P | same losses; 1 fp32 ulp in 39 weights |
| ZeRO-2 | 2.02 MiB | 2.4375 | 27,096 | 1.94 P | same |
| ZeRO-3 | 0.41 MiB | 0.5 | 27,096 | 2.91 P | same |

P is one bf16 copy of the weights (1,734,144 bytes here). The course rounds 1.94 P to 2P and 2.91 P to 3P; the communication table further down explains the gap.

## What the 32 GPUs are

A virtual GPU is a Python object (`VirtualGPU` in [`zero_sim/cluster.py`](zero_sim/cluster.py)) with a rank number from 0 to 31 and a dictionary of real PyTorch tensors that it keeps between steps. The dtypes are the ones from the notes' 16-byte table: bf16 weights (2 B), bf16 gradients (2 B), an fp32 master copy (4 B), Adam `m` (4 B) and Adam `v` (4 B). Because the tensors really have those dtypes, a rank's memory is just the sum of `tensor.nbytes` over its dictionary. I did not have to model the bytes separately; I count them.

The ranks do not share memory. They only exchange data through three collectives I wrote as ring algorithms ([`zero_sim/collectives.py`](zero_sim/collectives.py)): reduce-scatter, all-gather and all-reduce. Every ring step, each rank sends one chunk to its right-hand neighbour, and every send is logged with its size. All communication numbers below come from that log.

CPU threads would have been an option. I used plain objects run in lockstep instead: all 32 ranks do unit 1, then all do unit 2, and so on, meeting at each collective the way SPMD code does on a real cluster. Threads would only add scheduling nondeterminism, and none of what I wanted to measure (who holds what, what is sent, what the update is) depends on wall-clock overlap.

## The model

A 4-layer GPT: d_model 128, 4 heads, context 64, byte-level vocabulary of 256, untied output head. **867,072 parameters.** It trains on a short text I wrote (`data/corpus.txt`), 2 sequences of 64 bytes per rank, so the global batch is 64 sequences.

I chose the size from memory, not from a target. DP really allocates 32 complete copies of the 16-byte state, which at this size is 32 x 16 B x 867,072 = 423 MiB. My laptop has 7.8 GiB with not much free, and the notebook process's peak working set, read right after the DP run, was roughly 0.9 to 1.1 GiB (882 to 1,119 MiB across my runs; it depends on what else the machine is doing). Something like 3M parameters would have needed 1.43 GiB of replicas just for DP.

The model is split into six units: the embedding, four transformer blocks, and the final norm plus head. Each unit's parameters sit in one flat vector, and that vector is what gets split into 32 shards. Every unit size divides by 32 exactly (blocks are 198,272 = 32 x 6,196), so there is no padding and no rounding in the byte counts.

## What rank 11 stores

I picked rank 11 because it is an ordinary rank. Rank 0 is often special in real code (logging, checkpointing), and I did not want that to colour anything.

| tensor | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| bf16 weights | full | full | full | 1/32 |
| bf16 grads | full | full | 1/32 | 1/32 |
| fp32 master | full | 1/32 | 1/32 | 1/32 |
| Adam m | full | 1/32 | 1/32 | 1/32 |
| Adam v | full | 1/32 | 1/32 | 1/32 |
| **bytes** | 13,873,152 | 3,793,440 | 2,113,488 | 433,536 |

"1/32" is a specific range. In each transformer block rank 11 owns flat elements 68,156 to 74,351, which all land inside `mlp.fc.weight` (parts of rows 12 to 60 of the 512 x 128 matrix). In the embedding it owns rows 110 to 119 of the token table. With byte tokens those rows are the letters `n` to `w`, so under ZeRO-1 rank 11 is the only rank that holds Adam state for the embeddings of `n` through `w`, and after each update the other 31 ranks get the new `n` to `w` rows from rank 11 through the all-gather. The full list is in `results/summary.json` under `rank11_owned_slices`.

`figures/ownership_grid.png` draws the same thing for all 32 ranks: a full square where a tensor is replicated, a diagonal where it is sharded. Counting coloured cells in one rank's row, weighted by bytes, gives the bytes-per-parameter numbers. The notebook asserts the counted bytes equal the notes' formulas exactly: 16, 4 + 12/N, 2 + 14/N and 16/N.

## The collectives

Before any ZeRO stage, I checked the identity everything else rests on. Each of the 32 ranks computed the gradient of its own micro-batch at the initial weights, giving 32 different vectors of 867,072 fp32 numbers. I combined them two ways. One was a naive all-reduce where every rank receives every other rank's whole vector and adds. The other was my ring reduce-scatter followed by ring all-gather.

* Largest difference between the two results: **3.8e-6** on values up to 23.8, a relative error of about 1.6e-7. Both are equally close to a float64 sum, so the gap is only fp32 addition order.
* All 32 ranks ended with the identical vector.
* Sent per rank: the ring sent **1.94 x** the buffer size; the naive version sent **31 x**.

What made the rest click for me was looking at the intermediate state. After the reduce-scatter, rank 11 holds the fully averaged gradient for its own 1/32 slice and nothing else. That is exactly what it needs to update its 1/32 of the optimizer state. So the half-finished all-reduce is not a waste product. ZeRO stops there for gradients and then uses the all-gather half for something else.

## DP baseline

Every rank keeps everything. In one step each rank runs forward and backward on its own 2 sequences, the bf16 gradients are averaged with a ring all-reduce, the gradient is clipped to norm 1.0, Adam updates the full fp32 master, and the master is cast back to bf16. Three steps.

* After 3 steps the 32 replicas were **bitwise identical** in all five tensors. They start identical and apply the identical averaged gradient, so they stay identical.
* A separate single-device run, one "big GPU" with all 64 sequences and plain autograd, gave the same step-1 loss: 5.570058 for the 32 ranks, 5.570057 for the single device, a difference of 5.5e-7.
* In fp32, the average of the 32 micro-batch gradients matched the single full-batch gradient to a relative error of 1.1e-7. That is the notes' point that DP is the same training as one larger batch.
* The DP gradient, which is averaged in bf16 on the wire, had a relative error of 5.6e-3 against the fp32 answer. One bf16 rounding is about 2e-3, and the ring re-rounds the partial sum at each of its 31 hops. This is a cost of the bf16 wire format and applies equally to every stage.

The waste is easy to see in the numbers. Each rank ran Adam on all 867,072 parameters, so the cluster did 27.7 million Adam updates to change 867,072 weights. And the 32 ranks together held 423 MiB to store 13.2 MiB of information.

## ZeRO-1: the optimizer state goes

Rank 11's fp32 master, `m` and `v` dropped from 3,468,288 bytes each to 108,384 each. Its bf16 weights and gradient buffer did not change. Total: 13.23 MiB down to 3.62 MiB, a 3.66x reduction from sharding the 12 bytes that are only touched once per step.

The step changes in two places:

1. Instead of an all-reduce, the gradients go through a **reduce-scatter**. Rank 11 receives the averaged gradient for its slice and writes it into its own region of its gradient buffer. The other 31/32 of that buffer still holds rank 11's local, unaveraged gradient, which nobody reads. That is the waste ZeRO-2 removes.
2. Rank 11 runs Adam on its 27,096 elements only. Then an **all-gather** of the new bf16 weight shards gives every rank the full updated model again.

Communication per rank was the same as DP, 1.94 P. The difference is what the second half carries. In DP the all-gather half moves averaged gradients. In ZeRO-1 it moves updated weights. Same size, different payload, which is why the notes say ZeRO-1 is free in communication.

After the step, the 32 copies of the bf16 weights were bitwise identical again.

## ZeRO-2: the gradient buffer goes too

The only change from ZeRO-1 is rank 11's gradient storage: 1,734,144 bytes down to 54,192. As soon as one unit's backward finishes, its bf16 gradient becomes a temporary bucket. The bucket is reduce-scattered and rank 11 keeps its averaged slice. Then the bucket is freed. The largest temporary buffer rank 11 ever held was one block's bucket, 396,544 bytes, and never the whole model's gradient.

Total for rank 11: 2.02 MiB, 6.6x less than DP, still at 1.94 P of communication. The bf16 weights (1.65 MiB) are now 82% of what rank 11 keeps, and they are the only thing still replicated.

## ZeRO-3: the weights go, and have to be fetched

Rank 11 keeps 54,192 bytes of bf16 weights, 1/32 of the model. To run a block it needs the whole block, so the step becomes:

```
forward,  each unit:  all-gather its weights -> compute -> free them
backward, each unit:  all-gather its weights again -> compute grads -> reduce-scatter grads -> free both
update:               rank 11 updates its own shard; no all-gather needed afterwards
```

That is 12 gathers per step. Between the end of forward and the start of backward, rank 11 held **54,192 bytes of weights** where every other stage holds 1,734,144. Its worst moment in the step was 793,088 bytes of temporary buffers: one block's gathered weights plus that block's gradient bucket. That is still less than the full weight copy ZeRO-2 keeps permanently. `figures/zero3_gather_release_timeline.png` shows the sawtooth.

The release is real in the code, not just in a ledger, though my first version only got it half right. Normally autograd keeps a reference to each weight it needs for backward, which would keep the gathered block alive. I shrank the gathered tensor's storage to zero bytes after each unit's forward (the engine asserts every unit's storage is empty before backward starts) and replaced autograd's references to it with placeholders using `torch.autograd.graph.saved_tensors_hooks`. In backward the placeholders are filled from the second gather. I tested this on a small layer first and got gradients bitwise identical to normal autograd. Freeing the storage by resizing it to zero is the same trick FSDP uses; the hooks are my way of refilling it, because my gathered tensors are new objects each time. It meant I did not need recomputation, so ZeRO-3's FLOP count stays equal to DP's. A review pass then found that a leftover Python variable still pointed at each gathered bf16 block until the next gather, so the process briefly held one extra unit even though my ledger said it had been freed. I fixed that, and a test now uses weak references to check that every gathered block is actually garbage before the next one is gathered.

Communication went up to **2.91 P**: one P for the forward gathers, one for the backward gathers, one for the gradient reduce-scatter. The all-gather after the update disappears, because each rank only stores its own shard.

## Did the update stay the same?

I compared the fp32 master weights (rebuilt from the 32 shards for inspection) against DP:

| | avg gradient at step 1 | max diff after step 1 | max diff after step 3 | weights that differ | losses, all 3 steps |
|---|---|---|---|---|---|
| ZeRO-1 | bitwise equal | 0 | 1.86e-9 | 39 of 867,072 | identical |
| ZeRO-2 | bitwise equal | 0 | 1.86e-9 | 39 | identical |
| ZeRO-3 | bitwise equal | 0 | 1.86e-9 | 39 | identical |

I expected all zeros, so I chased the 1.86e-9. It is exactly one fp32 ulp for weights between 1/64 and 1/32, which is where these weights live. It is not communication: the step-1 averaged gradients are bitwise equal in all four stages. I also checked the gradient clipping. DP computes the gradient norm from one full gradient, while ZeRO ranks add 32 partial sums, and the two clip coefficients really do differ in double precision (0.12404414538705 vs 0.12404413969880 at step 1). But they multiply an fp32 gradient, and at all three steps both coefficients round to the same fp32 number, so the clipped gradients were bitwise identical (`results/clip_coefficients.csv`). Rerunning with clipping off gave the same 1-ulp result (44 weights), which fits. The cause turned out to be Adam itself. Applying the same Adam update to one whole vector and to the same vector cut into 32 slices, with no ranks or communication involved, gives exact agreement after step 1, then 35 elements 1 ulp apart at step 2 and 65 elements up to 2 ulp apart at step 3. Every one of those elements sits in the last (length mod 16) positions of a slice, the leftover after PyTorch's vectorised loop. A block slice is 6,196 long, remainder 4; the embedding slices are 1,280, remainder 0, and never differ. So the vectorised loop and its scalar tail round slightly differently, and a whole tensor and a slice put different elements in the tail.

None of this reached the bf16 weights the model actually computes with: after 3 steps they were bitwise identical in all four stages, and the losses matched at every step. The conclusion I take from it: the stages move state and change what is communicated, and the only trace they leave on the update is floating-point evaluation order.

One real surprise from development: before I replaced PyTorch's fused attention with explicit matmuls (so the FLOP counter could see the attention products), the same 3-step comparison gave a ZeRO-vs-DP gap of 4.0e-6 across about 78,000 weights, and 1.9e-9 with clipping off. So in that version clipping did matter. My best explanation is that at some step the two clip coefficients rounded to different fp32 values, which nudged some fp32 master weights across a bf16 rounding boundary, after which the forward passes differed and the gap grew. I did not prove that; the final code no longer shows it. The lesson I keep is that "same update" means "same up to summation order", and in mixed precision summation order can occasionally show up as more than one ulp.

## What "computation changes" means

The question here is how computation changes. This is the part I most wanted to get right, because ZeRO does not make the model cheaper to run.

* **Model compute is unchanged.** PyTorch's FLOP counter on rank 11 read 0.2265 GFLOP forward and 0.4530 GFLOP backward in every stage. The forward number also matches my hand count of the matmuls exactly (226,492,416).
* **Optimizer work is what gets divided.** Per rank, Adam went from 867,072 elements (DP) to 27,096 (all ZeRO stages). Across the cluster, from 27.7 million updates to 867,072, which is the actual number of parameters. ZeRO removes 31 redundant copies of the same arithmetic.
* **The gradient norm** for clipping is computed 32 times on full gradients under DP. Under ZeRO each rank square-sums only its shard, then one 4-byte all-reduce combines them.
* **Communication** is where ZeRO-3 pays:

| per step, per rank | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| measured, units of P | 1.9375 | 1.9375 | 1.9375 | 2.906 |
| ring formula 2(N-1)/N or 3(N-1)/N | 1.9375 | 1.9375 | 1.9375 | 2.906 |
| course value | 2P | 2P | 2P | 3P |
| collective calls | 6 | 13 | 13 | 19 |
| point-to-point sends, all ranks | 11,904 | 11,936 | 11,936 | 17,888 |

The course's 2P is the large-N limit. Each ring phase moves (N-1)/N of the buffer, so at N = 32 the exact value is 1.9375 P. DP's 6 all-reduces are 12 ring phases internally, so DP, ZeRO-1 and ZeRO-2 all do 12 large phases per step. ZeRO-3 does 18, and 12 of them are gathers that the forward or backward pass has to wait for. On real hardware that matters beyond the byte count, since each call has a fixed start-up cost and the gathers have to be prefetched and overlapped with compute (notes, section 10).

The simulator's wall times (DP between 10 and 22 s across my runs, the ZeRO stages about 4 to 6 s) say nothing about GPUs. The DP run is slow because it allocates and updates 423 MiB of replicas on a laptop, not because of anything a cluster would see.

## More GPUs do not fix DP

`figures/world_size_scaling.png` plots bytes per parameter per rank from N = 1 to 64, built from the same rank objects:

* DP stays at 16 B whatever N is, because every new rank brings a full copy.
* ZeRO-1 bends toward 4 B: the weights and gradients are still replicated.
* ZeRO-2 bends toward 2 B: the weights are still replicated.
* ZeRO-3 keeps falling as 16/N.

Those floors are the whole story of the 30B example.

## The same arithmetic at 30B

I read the fraction of each tensor a rank holds off our rank objects and multiplied by 30 billion parameters. Those fractions are exactly 1 or 1/N by construction, so this is the course formula evaluated through my ownership table, not an independent measurement; the point is that the same code reproduces the notes' numbers. The notes' ladder is only used as the thing to compare against:

| GiB per GPU | N = 8 | N = 16 | N = 32 | N = 64 | fits a 74.5 GiB card at N = 32? |
|---|---|---|---|---|---|
| DP | 447.0 | 447.0 | 447.0 | 447.0 | no, never |
| ZeRO-1 | 153.7 | 132.7 | 122.2 | 117.0 | no, never |
| ZeRO-2 | 104.8 | 80.3 | 68.1 | 62.0 | yes, from 32 GPUs |
| ZeRO-3 | 55.9 | 27.9 | 14.0 | 7.0 | yes, from 8 GPUs |

Every entry matches the notes to 0.05 GiB (asserted). An "80 GB" card is 74.5 GiB. ZeRO-1 can never fit because the replicated weights and gradients alone are 4 B x 30B = 111.8 GiB. That 4-byte floor fills a 74.5 GiB card at exactly 20 billion parameters, the boundary the notes mention.

This is why the toy model is enough. Nothing in the accounting depends on the model's size; the shard fractions are the same at 867 thousand parameters and at 30 billion.

## Activations are extra

All of the above is model state. Activations saved for backward are separate, set by batch size and sequence length, and identical in every stage. On rank 11 they came to 4.84 MiB (fp32 in my simulator; about half that in bf16). That is about a third of DP's state, but 1.3x ZeRO-1's entire state, 2.4x ZeRO-2's and 11.7x ZeRO-3's. Even at roughly half the size in bf16 they would exceed ZeRO-2's and ZeRO-3's state. So a stage that "fits" on the state table can still run out of memory once activations are added, and choosing a higher ZeRO stage does nothing about activations. Recomputation does.

## Break it

To see what each collective is for, I removed one at a time for a single step:

* **DP without the gradient all-reduce:** each replica stepped on its own 2 sequences, and after one step the replicas' weights differed by up to 0.0024.
* **ZeRO-1 without the weight all-gather:** each rank had updated only its own 1/32 of the weights and held stale values for the rest. Replicas differed by up to 0.0015.
* **ZeRO-3 gathering from the wrong owner:** the forward pass ran through a scrambled model. Loss 5.5445 instead of 5.5701.

## Pros and cons, and which one I would pick

| | good | bad |
|---|---|---|
| DP | simplest; one all-reduce per bucket; no gathers | 16 B/param on every GPU no matter how many GPUs; optimizer work done N times |
| ZeRO-1 | removes 12 of 16 bytes from replication at no extra communication; still simple | stuck at 4 B/param, so the model's weights and grads must fit on one GPU |
| ZeRO-2 | also removes the replicated gradients, still no extra communication | stuck at 2 B/param; gradient buckets have to be reduced during backward |
| ZeRO-3 | the only stage whose memory keeps falling with N; model never fully on one GPU | +50% communication; a gather before every unit in forward and backward, so it needs prefetch/overlap to be fast; more moving parts |

How I would choose: take the lowest stage that fits with room for activations, because each step up adds complexity and ZeRO-3 adds traffic. For the notes' 30B model on 32 GPUs I would pick ZeRO-2. It fits at 68.1 GiB, sends no more than DP, and avoids the gathers. In the notes' 64-GPU example, 2P over InfiniBand is already 34% of the compute time on H100 and 77% on B200, and ZeRO-3 would turn that 2P into 3P. But 68.1 GiB leaves only about 6 GiB for activations, so if a real step with activations does not fit, ZeRO-3 is the fallback. On 8 GPUs there is no choice: ZeRO-2 needs 104.8 GiB, and only ZeRO-3 (55.9 GiB) fits. That matches the open question in the notes: ZeRO-2 on 32 or ZeRO-3 on 8, decided by measured step time with activations included.

## How my sharding differs from the picture in class

In class the shards were described as contiguous groups of layers, so that GPU 1 holds the optimizer state for the first two layers, and so on. I shard every unit 32 ways instead, so rank 11 owns a slice of every block. For memory that makes no difference: each rank holds 1/32 either way. For ZeRO-3 it does matter. If rank 11 owned all of block 2, gathering block 2 would mean rank 11 alone sends the whole block to 31 ranks. Splitting each unit across everyone turns that into a balanced all-gather, where every rank sends the same amount. This is what FSDP does. DeepSpeed's stage 1/2 code flattens each parameter group into one buffer and cuts it into N contiguous pieces (`get_data_parallel_partitions` in `stage_1_and_2.py`), which is closer to the picture from class.

## What this simulator does and does not show

It demonstrates exactly: which rank holds which bytes, the collective identities, that the four stages produce the same update up to floating-point order, the communication volume under a ring model, and the state arithmetic that extrapolates to 30B.

It does not show anything about speed. CPU wall time is not GPU time, Python tensors are not the CUDA caching allocator (no fragmentation, no allocator overhead), and my ring has no bandwidth, latency, NVLink/InfiniBand topology or overlap with compute. Activations are counted in fp32. Reduce-scatter buckets are one unit, not a byte-size bucket. Real ZeRO-3 implementations also prefetch the next unit and can keep recently used weights gathered when memory allows (DeepSpeed's `stage3_prefetch_bucket_size`, `stage3_max_reuse_distance`); mine gathers every unit exactly twice.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace ZeRO_32_virtual_GPUs.ipynb
```

58 unit tests (a smaller model on 8 ranks for speed, plus 32-rank accounting) and 102 checks inside the notebook, all passing. The notebook writes every table to `results/` and every figure to `figures/`. Seeds are fixed (init seed 0, data seed 1), so reruns give the same numbers.

## Files

```
ZeRO_32_virtual_GPUs.ipynb    executed notebook (main deliverable)
zero_sim/model.py             the GPT, written as six units over flat parameter vectors
zero_sim/cluster.py           VirtualGPU, the stage -> sharding table, ownership helpers
zero_sim/collectives.py       ring reduce-scatter / all-gather / all-reduce with a send log
zero_sim/engine.py            one lockstep training step for all four stages, single-device reference
zero_sim/accounting.py        bytes per rank from rank objects, course formulas, 30B projection
zero_sim/plots.py             figures
tests/test_zero_sim.py        pytest suite
results/                      CSV/JSON evidence (summary.json, equivalence, memory, comm log, 30B ladder, assertions)
figures/                      ownership grid, memory, communication, scaling, ZeRO-3 timeline
data/corpus.txt               the training text
```

## References

* ERA V5 Session 12 course notes, sections 1, 3, 4, 6 and 7 (16-byte accounting, collectives, ZeRO stages, memory ladder).
* DeepSpeed ZeRO tutorial, stage definitions: <https://www.deepspeed.ai/tutorials/zero/>
* Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2019: <https://arxiv.org/abs/1910.02054>
