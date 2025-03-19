<h1 style="color:red;"><b>PLEASE, READ THIS BEFORE DOING ANYTHING !!!</b></h1>

1. Using **PhoBERT pre-trained model**
2. When build model, using **train+dev+test_set** from **processed** folder !!!
3. **Tokenizing:**: 
   - Set **max_length = 50**
   - Set **truncation = True**
   - Set **padding='max_length'**


<h1>Training Strategy Suggestion</h1> 
- Due to the time consuming and limited computing resource, here are suggest on training strategy:

1. Using special tokenizer specialized for BERT, if posible use phoBERT
2. Training Hyper Parameter: Due to small dataset.
- Use learning rate: 2e-5
- Warm_up steps if using steps: 10% of total steps.
- Batch-size: 32-64 (128 T4 can handle)
- epoch: 8-10 (use early stopping)
- use half-precision: fp16
- L2 regulation: 0.01 
- Optimizer: AdamW or ??? depend.
- for replace Grid_Search use Population-Based Training ??? (first time hear)

3. Some note: 
- use tradditional metric
- training BERT is time-consuming(with 3 epoch = 1.5 hour), prepare all plan before training.
- At very first steps and epoch: can archive very high accuracy.
- I used Step method instead of epoch. How ever the training still required epoch for calculate the number of steps.


