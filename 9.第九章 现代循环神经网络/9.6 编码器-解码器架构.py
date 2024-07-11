# 9.6 编码器-解码器架构
from torch import nn


# 编码器
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


# 解码器
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        # 用来接收编码器的输出，转换成state
        raise NotImplementedError

    def forward(self, X, state):
        # 且在forward中解码器也有自己的输入
        raise NotImplementedError


# 编码器合并解码器
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):  # enc_X：编码器输入，dec_X：解码器输入
        enc_outputs = self.encoder(enc_X, *args)  # enc_outputs：经过编码器得到编码器输出
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 通过解码器的init_state方法将编码器的输出变成一个状态供解码器使用
        return self.decoder(dec_X, dec_state)
        # 根据刚刚得到的状态和解码器自己的输入，得到解码器最终的输出








