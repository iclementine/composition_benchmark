from transformer import *
import paddle
import time

FEATURE_SIZE = 512
BATCH_SIZE = 4
LENGTH = 580
LAYERS = 4
HEADS = 4

DILATION_FACTOR = 6.4
DECODER_INPUT_SIZE = 80
REDUCTION_FACTOR = 4
STEPS = 100

model = Transformer(99, FEATURE_SIZE, FEATURE_SIZE, DECODER_INPUT_SIZE, HEADS,
                    2 * FEATURE_SIZE, LAYERS, LAYERS, 32, 2 * FEATURE_SIZE, REDUCTION_FACTOR, 0.5, 0.1)
text = paddle.randint(1, 99, [BATCH_SIZE, LENGTH])
text_lengths = paddle.randint(LENGTH // 2, LENGTH, [BATCH_SIZE])
encoder_mask = paddle.nn.functional.sequence_mask(text_lengths, LENGTH)
text = text * encoder_mask

decoder_steps = int(DILATION_FACTOR * LENGTH) // REDUCTION_FACTOR * REDUCTION_FACTOR
target = paddle.randn([BATCH_SIZE, int(DILATION_FACTOR * LENGTH), DECODER_INPUT_SIZE])
mel = target[:, ::REDUCTION_FACTOR, :]

opt = paddle.optimizer.SGD(0.001, parameters=model.parameters())

total_time = 0
with paddle.no_grad():
    for _ in range(STEPS):
        start = time.time()
        outputs = model(text, mel)
        predicted = outputs["mel_output"]
        loss = paddle.nn.functional.mse_loss(predicted, target)

        # loss.backward()
        # opt.step()
        # opt.clear_gradients()

        end = time.time()
        ellapsed = end - start
        total_time += ellapsed
        print("time: ", ellapsed)
    print("average: ", 1000 * total_time / STEPS)
