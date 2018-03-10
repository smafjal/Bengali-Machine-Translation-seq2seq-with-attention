import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import utils
import models as model
from language import Language

target_language = "ben"
teacher_forcing_ratio = 0.5
clip = 3.0

# attn_model = 'concat'
attn_model = 'dot'
hidden_size = 500
n_layers = 4
dropout_p = 0.05
learning_rate = 0.000001

# Configuring training
n_epochs = 300
plot_every = 1
print_every = 1


def train(input_var, target_var, encoder, decoder, encoder_opt, decoder_opt, criterion):
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    loss = 0
    
    target_length = target_var.size()[0]
    
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_var, encoder_hidden)
    
    decoder_input = Variable(torch.LongTensor([0]))
    decoder_input = decoder_input
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_context = decoder_context
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden,
                                                                                         encoder_outputs)
            
            current_loss = criterion(decoder_output, target_var[di])
            loss += current_loss  # criterion(decoder_output, target_var[di])
            decoder_input = target_var[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden,
                                                                                         encoder_outputs)
            
            current_loss = criterion(decoder_output, target_var[di])
            loss += current_loss  # criterion(decoder_output, target_var[di])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input
            if ni == Language.eos_token:
                break
        
    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_opt.step()
    decoder_opt.step()
    
    return loss.data[0] / target_length


def save_model(mymodel, path):
    torch.save(mymodel.state_dict(), path)


def load_model(path):
    return torch.load(path)


def main():
    input_lang, output_lang, pairs = utils.prepare_data(lang_name=target_language, _dir='data')
    
    encoder = model.EncoderRNN(input_lang.n_words, hidden_size, n_layers)
    decoder = model.AttentionDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)
    
    print("Encoder-Model: ", encoder)
    print("Decoder-Model: ", decoder)
    
    # Initialize optimizers and criterion
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    # Begin training
    for epoch in range(1, n_epochs + 1):
        
        training_pair = utils.variables_from_pair(random.choice(pairs), input_lang, output_lang)
        input_variable = training_pair[0]
        target_variable = training_pair[1]
        
        # Run the train step
        epoch_loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        print_loss_total += epoch_loss
        plot_loss_total += epoch_loss
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            time_since = utils.time_since(start, epoch / n_epochs)
            print('%s (%d %d%%) %.4f' % (time_since, epoch, epoch / n_epochs * 100, print_loss_avg))
        
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    save_model(encoder, 'data/encoder_state_' + target_language)
    save_model(decoder, 'data/decoder_state_' + target_language)
    save_model(decoder.attention, 'data/decoder_attention_state_' + target_language)
    utils.show_plot(plot_losses)


if __name__ == "__main__":
    main()
