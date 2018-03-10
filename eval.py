import torch
from torch.autograd import Variable

import utils
import models as model
from language import Language

# Parse argument for input sentence

language = 'ben'
input_lang, output_lang, pairs = utils.prepare_data(language, _dir='data')
attn_model = 'dot'
hidden_size = 500
n_layers = 4
dropout_p = 0.05

# Initialize models
encoder = model.EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = model.AttentionDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

print("Load parameters")

# Load model parameters
encoder.load_state_dict(torch.load('data/encoder_state_{}'.format(language)))
decoder.load_state_dict(torch.load('data/decoder_state_{}'.format(language)))
decoder.attention.load_state_dict(torch.load('data/decoder_attention_state_{}'.format(language)))


def evaluate(sentence, max_length=10):
    input_variable = utils.variable_from_sentence(input_lang, sentence)

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[Language.sos_token]]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_input = decoder_input
    decoder_context = decoder_context
    
    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data
        
        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == Language.eos_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input
    
    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def main():
    user_input = 'Who cares?'
    sentence = utils.normalize_string(user_input)
    output_words, decoder_attn = evaluate(sentence)
    output_sentence = ' '.join(output_words)
    print("Sentence: {}\nTranslated Sentence: {}".format(user_input, output_sentence))


if __name__ == "__main__":
    main()
