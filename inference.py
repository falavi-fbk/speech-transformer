#!/datadrive/speechtek/daniele/anaconda3/bin/python
## Usage


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
from typing import Optional, Any, Union, Callable
import numpy as np
import random
import sys
import io
import editdistance
import re

from torchaudio.models.decoder import ctc_decoder
from ctc_decoder import BKTree
from ctc_decoder import LanguageModel
from ctc_decoder import beam_search
from ctc_decoder import best_path

#from ctc_decoder import loss
from ctc_decoder import prefix_search_heuristic_split
from ctc_decoder import probability
from ctc_decoder import token_passing     
#from ctc_decoder import lexicon_decoder


from torchtext.vocab import vocab
import math
import os
import torchaudio
from torch.optim import Adam
from torch.autograd import Variable
from typing import Tuple
from torch.utils import data

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer


cuda = torch.cuda.is_available()  
#device = torch.device('cuda' if cuda else 'cpu')
device = 'cpu'
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
            from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        print("RATE:",rate)
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))) 


class Conv1dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)
    Args:
        in_channels (int): Number of channels in the input image                                             
        out_channels (int): Number of channels produced by the convolution                       
                                                                                                                
    Inputs: inputs                                                       
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs
    Returns: outputs, output_lengths 
        - **outputs** (batch, time, dim): Tensor produced by the convolution

   """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv1dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros'),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros')
        )
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = self.sequential(inputs)
        return outputs

class _PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class PositionalEncoding(nn.Module):
    def __init__(self,  d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """     
            Args:
              x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x=x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1,0,2)
        return self.dropout(x)
    
        
class MyTransformer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            norm_first: bool = False,
            features_length = 80,
            num_heads: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 3,
            dim_feedforward = 1024,
            n_phones: int = 32,
            dropout_f: float = 0.1,
            batch_first: bool = True,
      ) -> None:
#        super(MyTransformer, self).__init__()
        super().__init__()
        self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=dropout_f, max_len=2000)
        self.layer_norm=nn.LayerNorm(d_model,eps=1e-5) 
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout_f, norm_first=norm_first, batch_first=batch_first),  num_encoder_layers,self.layer_norm)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout_f, batch_first=batch_first, norm_first = norm_first),  num_decoder_layers, self.layer_norm) 
#        self.my_model= nn.Transformer(dropout=dropout_f, norm_first=norm_first, d_model=d_model, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=batch_first)
        self.generate_square_subsequent_mask = nn.Transformer.generate_square_subsequent_mask 
        self.linear_enc=nn.Linear(d_model, n_phones)
        self.linear_dec=nn.Linear(d_model, n_phones)        
        self.softmax=nn.Softmax(dim=2)
        self.relu=nn.ReLU()
        self.embedding = nn.Embedding(n_phones, d_model) 

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
            # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
            # [False, False, False, True, True, True]
            return (matrix == pad_token)
        
    def create_tgt_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    
    def _encoder_(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src = self.conv_subsample(src.permute(0,2,1)).permute(0,2,1)

        src = src * math.sqrt(d_model)
        src = self.positional_encoder(src)

        encoder_out=self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        #encoder_out=self.linear(encoder_out)
        #encoder_out=torch.nn.functional.log_softmax(encoder_out,dim=2)    

        return encoder_out        

    def _decoder_(self, tgt: Tensor, encoder_out: Tensor, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt = self.embedding(tgt) * math.sqrt(d_model)
        tgt = self.positional_encoder(tgt)
        decoder_out=self.decoder(tgt,encoder_out,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        decoder_out=self.linear_dec(decoder_out)         
        decoder_out=torch.nn.functional.log_softmax(decoder_out,dim=2)    
        return decoder_out
    
    def _conv_subsample_(self, src: Tensor) -> Tensor:
        return self.conv_subsample(src).permute(0,2,1)
    
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:


        # Src size must be (batch_size, src sequence length)
        
        # Tgt size must be (batch_size, tgt sequence length)

        #print("SRCLEN_:",src)
        src = self.conv_subsample(src.permute(0,2,1)).permute(0,2,1)
        #print("SRC:",src)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)

        #src = self.embedding(src) * math.sqrt(d_model)
        src = src * math.sqrt(d_model)
        tgt = self.embedding(tgt) * math.sqrt(d_model)
        
        tgt = self.positional_encoder(tgt)
        src = self.positional_encoder(src)
        #print("TGT_MASK:",tgt_mask)
        #print("TGT_KEY_MASK:",tgt_key_padding_mask)


        encoder_out=self.encoder(src,src_key_padding_mask=src_key_padding_mask)
        decoder_out=self.decoder(tgt,encoder_out,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        
        decoder_out=self.linear_dec(decoder_out)
        decoder_out=torch.nn.functional.log_softmax(decoder_out,dim=2)    
        
        encoder_out=self.linear_enc(encoder_out)
        encoder_out=torch.nn.functional.log_softmax(encoder_out,dim=2)    

        #transformer_out=self.my_model(src,tgt,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        #out=self.linear(transformer_out)
        #out=self.relu(out)        
        #out=self.softmax(out)
        return decoder_out, encoder_out



# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

            # compute the accuracy over all test images
            accuracy = (100 * accuracy / total)
            return(accuracy)

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        # 30
        ^ 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        ' 29
        $ 31
        @ 0
        """
# ^=<SOS> 1
# $=<EOS> 31
# #=<PAD> 30
# @=<blank> for ctc
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()

            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

        self.index_map[28] = ' '
        
    def text_to_int(self,text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = 28#self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i.detach().item()])
        return ''.join(string)#.replace('', ' ')
    
sample_rate = 16000
n_fft = 512
win_length = 320 #20ms
hop_length = 160 #10ms
n_mels = 80
n_mfcc = 80

dim_feedforward=2048#1024

torch.set_printoptions(profile="full")
os.makedirs('./model/best', exist_ok=True)
os.makedirs('./model/worst', exist_ok=True) 
import torchaudio.transforms as T

mfcc_transform = T.MFCC(sample_rate=sample_rate, \
                        n_mfcc=n_mfcc, \
                        melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'win_length': win_length, 'hop_length': hop_length})

spec_transform = T.Spectrogram(n_fft=n_fft * 2, \
                               hop_length=hop_length, \
                               win_length=win_length)

n_phones=32
d_model=512 #256
num_heads=8
num_encoder_layers=12
num_decoder_layers=6
model = MyTransformer(n_phones=n_phones, \
                      d_model=d_model, \
                      features_length=n_mfcc, \
                      num_heads=num_heads, \
                      norm_first=True, \
                      num_encoder_layers=num_encoder_layers, \
                      dim_feedforward=dim_feedforward, \
                      num_decoder_layers=num_decoder_layers, \
                      batch_first=True).to(device)

loss_fn = nn.CrossEntropyLoss()
ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True) 
warmup=16000
opt = NoamOpt(d_model, warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

text_transform=TextTransform()

def pad_sequence(batch, padvalue):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=padvalue)
    return batch.permute(0, 2, 1)

def ins_eos(inpseq, EOS_token=31):
    targets = torch.LongTensor(inpseq.size(0),1,inpseq.size(2)+1)
    for k in range(0,inpseq.size(0)):
        targets[k][0][0:inpseq.size(2)]=inpseq[k][0]
        targets[k][0][inpseq.size(2)]=EOS_token
    return targets      




def _collate_fn_(batch, SOS_token=1, EOS_token=31, PAD_token=30):

# A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets, source_pad = [], [], []

    # Gather in lists, and encode labels as indices
    t_len=torch.LongTensor(len(batch),1)
    k=0
    for waveform, _, label, *_ in batch: #spkid, cha_id, ut_id in batch:
        

        spec=mfcc_transform(waveform).to(device)
        spec = model._conv_subsample_(spec).permute(0,2,1)
        s_pad=torch.ones(spec.size(2))
        source_pad += [s_pad.unsqueeze(0)]
        
        tensors += spec
        del spec
        del s_pad
        #label=label[10:110]
        tg=torch.LongTensor(text_transform.text_to_int("^"+label.lower()+"$"))
        targets += [tg.unsqueeze(0)]
        t_len[k][0]=len(tg)
        k=k+1
        del waveform
        del label

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors,0)
    targets = pad_sequence(targets,PAD_token)
    source_pad = pad_sequence(source_pad,0)
#    targets = ins_eos(targets, EOS_token=EOS_token)
    return tensors.squeeze(1), targets.squeeze(1), source_pad.squeeze(1), t_len

def collate_fn(batch, SOS_token=1, EOS_token=31, PAD_token=30):

# A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets, source_pad = [], [], []

    # Gather in lists, and encode labels as indices
    t_len=torch.LongTensor(len(batch),1)
    k=0
    for waveform, _, label, *_ in batch: #spkid, cha_id, ut_id in batch:
        

        spec=mfcc_transform(waveform).to(device)
        pad_spec=torch.zeros(spec.size(0),spec.size(1), 800).to(device)
        #pad_spec=torch.zeros(spec.size(0),spec.size(1), 3600-spec.size(2)).to(device)
        spec=torch.cat((spec,pad_spec),2)
        tensors += spec


        del pad_spec
        del spec
        
        #label=label[10:110]

        tg=torch.LongTensor(text_transform.text_to_int("^"+label.lower()+"$"))
        pad_tg=torch.full((620 - tg.size(0),),PAD_token)
        tg=torch.cat((tg,pad_tg),0)
        del pad_tg
        targets += [tg.unsqueeze(0)]
        t_len[k][0]=len(tg)
        k=k+1
        del waveform
        del label

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors,0)
    targets = pad_sequence(targets,PAD_token)
    return tensors.squeeze(1), targets.squeeze(1), t_len


#from ctc_decoder import download_pretrained_files

#files = download_pretrained_files("librispeech-4-gram")

#print(files)



LM_WEIGHT = 3.23
WORD_SCORE = -0.26
N_BEST = 30
#beam_search_decoder = lexicon_decoder(
beam_search_decoder = ctc_decoder(
        lexicon="lexicon.txt",
        tokens="tokens.txt",
        lm="lm.bin",
#        lm="",
        nbest=N_BEST,
        beam_size=1500,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE
    )

#beam_search_transformer_decoder = lexicon_decoder(
'''
beam_search_transformer_decoder = ctc_decoder(
        lexicon="transf.tokens",
        tokens="tokens.txt",
        lm="",
        nbest=1,
        beam_size=10,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE
    )
'''
def predict(model, input_sequence, w_lex, max_length=80, SOS_token=1, EOS_token=31, PAD_token = 30):
        """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        """
        y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
        encoder_out = model._encoder_(input_sequence)

        _,ctc_out = model(input_sequence, y_input)
        print(ctc_out.size())
        cout_t = ctc_out.data.topk(1)[1]
        ctc_out=apply_lex(re.sub(r"[#^$@]+","",text_transform.int_to_text(cout_t.squeeze(0))),w_lex)
        s_len=len(ctc_out)
#        print("COUT[",0,"]:",text_transform.int_to_text(cout_t[0]))  
        
        
        for _ in range(max_length):
            # Get source mask

            tgt_mask=model.create_tgt_mask(y_input.size(1)).to(device)
            tgt_pad_mask=model.create_pad_mask(y_input,30).to(device) 
            pred = model._decoder_(y_input, encoder_out = encoder_out, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_pad_mask)
            print(pred.size())
            del tgt_mask
            del tgt_pad_mask
            next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            next_item = torch.tensor([[next_item]], device=device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)
            
            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_token: # or next_item.view(-1).item() == PAD_token or  next_item.view(-1).item() == SOS_token:
                break
            del next_item
            del pred

#        print("Y_OUT:",text_transform.int_to_text(y_input.squeeze(0)).replace('#',''))
        del encoder_out
        return y_input, ctc_out#.view(-1).tolist()


def adjust_learning_rate(opt, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in opt.param_groups:
        param_group['lr'] = lr                                                                               
def load_dict(file_path):
    dict=[]
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            dict += [line.strip("\n")]
    return dict
def apply_lex(predicted, lexicon):
    lex_out=[]
    for w in predicted.split(" "):
        if w in lexicon:
            lex_out += [w]
        else:  
            min_lex=99999
            w_min=""
            for w_lex in lexicon:
                d_lex=editdistance.eval(w, w_lex)
                if d_lex < min_lex:
                    min_lex = d_lex
                    w_min = w_lex
            lex_out += [w_min]
                    
    return " ".join([str(item) for item in lex_out])


def beam_predict(model, input_sequence, w_lex, w_ctc, max_length=80, SOS_token=1, EOS_token=31, PAD_token = 30):
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)    
    _,ctc_out = model(input_sequence, y_input)
    
    del y_input
    encoder_out = model._encoder_(input_sequence)    
    beam_search_result = beam_search_decoder(ctc_out.cpu())

    n_hyps=len(beam_search_result[0])
    s_pred = torch.zeros(n_hyps)
    s_ctc = torch.zeros(n_hyps)    
    for i in range(0,n_hyps):
        beam_search_transcript = " ".join(beam_search_result[0][i].words).strip()
        #print("TRANSCRIPT:",beam_search_transcript)
        out_ctc=apply_lex(re.sub(r"[#^$@]+","",beam_search_transcript),w_lex)
        if i == 0:
            _ctc_ = out_ctc
            
        beam_predict="^"+out_ctc

        beam_p=torch.LongTensor(text_transform.text_to_int(beam_predict)).unsqueeze(0).to(device)

        #print("ENCODER_OUT:",encoder_out.size())
        tgt_mask=model.create_tgt_mask(beam_p.size(1)).to(device)
        tgt_pad_mask=model.create_pad_mask(beam_p,30).to(device)
        pred = model._decoder_(beam_p, encoder_out = encoder_out, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_pad_mask)
        #out_lex = apply_lex(re.sub(r"[#^$@]+","",text_transform.int_to_text(pred.data.topk(1)[1].squeeze(0))),w_lex)
        #print("PRED_TRANSCRIPT:",out_lex)
        #print("PRED_SIZE[",i,"]:",pred.size())
        #print("BEAM:",beam_search_result[0][i].tokens.size())
        #print("PRED_SUM_[",i,"]:",torch.sum(pred.data.topk(1)[0]).item())
        #print("BEAM:",beam_search_result[0][i])

        s_ctc[i] = beam_search_result[0][i].score / pred.size(1) #beam_search_result[0][i].tokens.size(0)

        s_pred[i] = torch.sum(pred.data.topk(1)[0]).item() 


        del beam_predict
        del beam_p
        del tgt_mask
        del tgt_pad_mask
        del pred
    max_v=-9999999999999999
    #s_pred=s_pred / -torch.max(s_pred)
    #s_ctc = s_ctc / -torch.max(s_ctc)
    for i in range(0,n_hyps):
        s_norm = s_ctc[i] * w_ctc + s_pred[i] * (1-w_ctc)
        #print("S_CTC:[",i,"]:",s_ctc[i].item())
        #print("S_PRED:[",i,"]:",s_pred[i].item())
        if(s_norm >  max_v):
            max_v = s_norm
            nbest_index = i
        
        
    beam_search_transcript = " ".join(beam_search_result[0][nbest_index].words).strip()
    ctc_out=apply_lex(re.sub(r"[#^$@]+","",beam_search_transcript),w_lex)        
    del beam_search_transcript

    del encoder_out
    del beam_search_result
    del s_pred
    del s_ctc
    return ctc_out, _ctc_


def _beam_predict_(model, input_sequence, w_lex, w_ctc, max_length=80, SOS_token=1, EOS_token=31, PAD_token = 30):
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)    
    _,ctc_out = model(input_sequence, y_input)
    
    del y_input
    encoder_out = model._encoder_(input_sequence)    
    beam_search_result = beam_search_decoder(ctc_out.cpu())

    n_hyps=len(beam_search_result[0])
    s_pred = torch.zeros(n_hyps)
    s_ctc = torch.zeros(n_hyps)    
    for i in range(0,n_hyps):
        beam_search_transcript = " ".join(beam_search_result[0][i].words).strip()
        #print("TRANSCRIPT:",beam_search_transcript)
        out_ctc=apply_lex(re.sub(r"[#^$@]+","",beam_search_transcript),w_lex)
        if i == 0:
            _ctc_ = out_ctc
            

        beam_predict="^"+out_ctc
        beam_p=torch.LongTensor(text_transform.text_to_int(beam_predict)).unsqueeze(0).to(device)

        beam_predict=out_ctc+"$"
        y_expected=torch.LongTensor(text_transform.text_to_int(beam_predict)).unsqueeze(0).to(device)        
        
        #print("ENCODER_OUT:",encoder_out.size())
        tgt_mask=model.create_tgt_mask(beam_p.size(1)).to(device)
        tgt_pad_mask=model.create_pad_mask(beam_p,30).to(device)
        pred = model._decoder_(beam_p, encoder_out = encoder_out, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_pad_mask)

        ctc_input_len=torch.full(size=(encoder_out.size(0),), fill_value = encoder_out.size(1), dtype=torch.long)
        ctc_target_len=torch.full(size=(1,1),fill_value=0, dtype=torch.long)
        ctc_target_len[0]=y_expected.size(1)

        #print(data[1],ctc_input_len,ctc_target_len)
        loss_ctc = ctc_loss(encoder_out.permute(1,0,2),y_expected,ctc_input_len,ctc_target_len).to(device)

        s_ctc[i] = loss_ctc.item() 
        s_pred[i] = torch.sum(pred.data.topk(1)[0]).item()
        #s_pred[i] = loss_fn(pred.permute(0,2,1), y_expected).item()
        
        print("s_ctc[",i,"]:",s_ctc[i].item())
        print("s_pred[",i,"]:",s_pred[i].item())        

        del beam_predict
        del beam_p
        del y_expected
        del tgt_mask
        del tgt_pad_mask
        del pred

        del loss_ctc
        
    max_v=-9999999999999999 #-9999999999999999
    #s_pred=s_pred / -torch.max(s_pred)
    #s_ctc = s_ctc / -torch.max(s_ctc)
    for i in range(0,n_hyps):
        s_norm = s_ctc[i] * w_ctc + s_pred[i] * (1-w_ctc)
        #print("S_CTC:[",i,"]:",s_ctc[i].item())
        #print("S_PRED:[",i,"]:",s_pred[i].item())
        if(s_norm >  max_v):
        #if(s_norm <  max_v):            
            max_v = s_norm
            nbest_index = i
        
        
    beam_search_transcript = " ".join(beam_search_result[0][nbest_index].words).strip()
    ctc_out=apply_lex(re.sub(r"[#^$@]+","",beam_search_transcript),w_lex)        
    del beam_search_transcript

    del encoder_out
    del beam_search_result
    del s_pred
    del s_ctc
    return ctc_out, _ctc_


def evaluate():
    batch_size=1
    file_dict='librispeech.lex'
    words=load_dict(file_dict)

    nepoch=68
    #best_model='{}/model/federated_base_noam_second/mod{:03d}-transformer'.format('.',nepoch)
    #best_model='{}/federated_models/srv_{:03d}-transformer'.format('.',nepoch)
    #best_model='{}/model/second_nheads08_nenclayers8/mod{:03d}-transformer'.format('.',nepoch)
    best_model='{}/model/all-d512/mod{:03d}-transformer'.format('.',nepoch)            
    model.load_state_dict(torch.load(best_model))
    model.to(device)
    w_ctc = float(sys.argv[1])
    #print("CTC_WEIGHT:",w_ctc)

    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print("TOT_PARAMS:",total_params)
    m1=model.state_dict()
    nc = 1

    for nepoch in range(69,89):
        #best_model='{}/federated_models/srv_{:03d}-transformer'.format('.',nepoch)
        #best_model='{}/model/second_nheads08_nenclayers8/mod{:03d}-transformer'.format('.',nepoch)        
        best_model='{}/model/all-d512/mod{:03d}-transformer'.format('.',nepoch)            
        if os.path.exists(best_model):
            print("Averaging with:", best_model)
            model.load_state_dict(torch.load(best_model))
            m2=model.state_dict()
            for key in m2:
                m1[key] = m2[key]+m1[key]
            nc = nc +1
            del m2
    
    for key in m1:
        m1[key] = m1[key] / nc
        
    model.load_state_dict(m1)
    del m1
    model.eval()
    #for set_ in "test-clean",:
    for set_ in "test-clean","dev-clean", "test-other", "dev-other":
        print(set_)
        test_dataset = torchaudio.datasets.LIBRISPEECH("/data/", url=set_, download=False)
        data_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=False, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn)

        for data in data_loader:
            y_expected = data[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
            for y_expected_ in y_expected:
                print(set_,"EXPECTED:",re.sub(r"[#^$]+","",text_transform.int_to_text(y_expected_.squeeze(0))))
                #print(set_,"EXPECTED:",text_transform.int_to_text(y_expected_.squeeze(0)))
                del y_expected_
                
            spec=data[0].transpose(1,2)

            for spec_ in spec:
                ctc_out, _ctc_ =beam_predict(model, spec_.unsqueeze(0), words, w_ctc,max_length=620)    
                print(set_,"MIX_OUT:", ctc_out)
                print(set_,"CTC_OUT:", _ctc_)                
                del spec_
                del ctc_out
                del _ctc_
            del data
            del y_expected
            del spec


def _evaluate_():
    batch_size=1
    file_dict='librispeech.lex'
    words=load_dict(file_dict)

    nepoch=290
    #best_model='{}/model/stek_ctc/mod{:02d}-transformer'.format('.',nepoch)
    best_model='{}/model/federated_models-1E/srv_{:3d}-transformer'.format('.',nepoch)    

    model.load_state_dict(torch.load(best_model)).to(device)
    
    m1=model.state_dict()
    nc = 1
    for nepoch in range(290,300):
        #best_model='{}/model/stek_ctc/mod{:02d}-transformer'.format('.',nepoch)
        best_model='{}/model/federated_models-1E/srv_{:3d}-transformer'.format('.',nepoch)            
        if os.path.exists(best_model):
            model.load_state_dict(torch.load(best_model))
            m2=model.state_dict()
            for key in m2:
                m1[key] = m2[key]+m1[key]
            nc = nc +1
    for key in m1:
        m1[key] = m1[key] / nc
    model.load_state_dict(m1)
    
#    nepoch=47
#    best_model='{}/model/best8/mod{:02d}-transformer'.format('.',nepoch)
#    model.load_state_dict(torch.load(best_model))
    model.eval()
    for set_ in "test-clean",  "dev-clean", "test-other", "dev-other":
        print(set_)
        test_dataset = torchaudio.datasets.LIBRISPEECH("/data/", url=set_, download=False)
        data_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=False, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn)


        for data in data_loader:
            y_expected = data[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
            for y_expected_ in y_expected:
                print(set_,"EXPECTED:",re.sub(r"[#^$]+","",text_transform.int_to_text(y_expected_.squeeze(0))))
                #print(set_,"EXPECTED:",text_transform.int_to_text(y_expected_.squeeze(0)))
                del y_expected_
                
            spec=data[0].transpose(1,2)

            for spec_ in spec:
                xtr="their piety would"
                out, ctc_out=predict(model, spec_.unsqueeze(0), words, max_length=620)    
#                print(set_,"OUT:",re.sub(r"[#^$]+","",text_transform.int_to_text(out.squeeze(0))))
#                print(set_,"OUT:",text_transform.int_to_text(out.squeeze(0)))
                out_lex = apply_lex(re.sub(r"[#^$]+","",text_transform.int_to_text(out.squeeze(0))),words)
#                out_ctc = apply_lex(re.sub(r"[#^$@]+","",text_transform.int_to_text(ctc_out.squeeze(0))),words)
                print(set_,"CTC_OUT:", ctc_out)
                print(set_,"OUT_LEX:",out_lex)
                del spec_
                del out
                del ctc_out

                del out_lex
            del data
            del y_expected
            del spec

#train(1000)
evaluate()
