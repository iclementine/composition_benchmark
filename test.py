from transformer import *
import paddle


layer = LayerNorm(8)
x = paddle.randn([3, 8])
y = layer(x)
y.backward()
print("layer norm")
exit(0)

layer = TransformerEncoderLayer(512, 16, 1024, 0.1)
x = paddle.randn((4, 280, 512))
encoder_mask = paddle.ones((4, 1, 280))
y, w = layer(x, encoder_mask)
print("encoder output shape: ", y.shape)
print("encoder attention shape: ", w.shape)
# w.sum().backward()



layer = TransformerDecoderLayer(256, 16, 512, 0.1, d_encoder=512)
x = paddle.randn((4, 380, 256))
decoder_mask = paddle.ones((4, 1, 380))

y, sw, cw = layer(x, y, y, encoder_mask, decoder_mask)
print("decoder output shape: ", y.shape)
print("decoder self attention shape: ", sw.shape)
print("decoder cross attention shape: ", cw.shape)

encoder = TransformerEncoder(512, 16, 1024, 8)
x = paddle.randn((4, 280, 512))
encoder_mask = paddle.ones((4, 1, 280))
y, ws = encoder(x, encoder_mask)
print("encoder output shape: ", y.shape)
for i, item in enumerate(ws):
    print(f"encoder attention shape {i}:", item.shape)

decoder = TransformerDecoder(256, 4, 512, 8, 0.1, d_encoder=512)
x = paddle.randn((4, 280, 512))
encoder_mask = paddle.ones((4, 1, 280))
decoder_mask = paddle.ones((4, 1, 380))
encoder_output, ws = encoder(x, encoder_mask)
q = paddle.randn((4, 380, 256))
decoder_out, sws, cws = decoder(q, encoder_output, encoder_output, encoder_mask, decoder_mask)
print("decoder output shape: ", decoder_out.shape)
for i, item in enumerate(sws):
    print(f"decoder self attention shape {i}:", item.shape)
for i, item in enumerate(cws):
    print(f"decoder cross attention shape {i}:", item.shape)


model = Transformer(99, 512, 256, 80, 4, 512, 4, 4, 32, 512, 4, 0.5, 0.1)
text = paddle.randint(1, 99, [4, 280])
text_lengths = paddle.randint(140, 280, [4])
encoder_mask = paddle.nn.functional.sequence_mask(text_lengths, 280)
text = text * encoder_mask
mel = paddle.randn([4, 800, 80])
mel_lengths = paddle.randint(680, 1037, [4])

outputs = model(text, mel)
