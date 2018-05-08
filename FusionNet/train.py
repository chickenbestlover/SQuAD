#coding: utf-8
import argparse
import torch
import ujson as json
from model import FusionNet
from utils.dataset import load_data, get_batches

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../SQuAD/')
parser.add_argument('--model_dir', default='train_model/batch256_dropout0.3_hidden125-250',
                    help = 'path to store saved models.')
parser.add_argument('--seed', default=8191)
parser.add_argument('--cuda', default=True,
                    help = 'whether to use GPU acceleration.')

### parameters ###
parser.add_argument('--epochs', type = int, default=50)
parser.add_argument('--eval', type = bool, default=False)
parser.add_argument('--batch_size', type = int, default=256)
parser.add_argument('--lrate', type = float, default=0.002)
parser.add_argument('--dropout', type = float, default=0.3)
parser.add_argument('--char_dim', type = int, default=50)
parser.add_argument('--pos_dim', type = int, default=12)
parser.add_argument('--ner_dim', type = int, default=8)
parser.add_argument('--evaluate', type = bool, default=False)
parser.add_argument('--char_hidden_size', type = int, default=50)
parser.add_argument('--hidden_size', type = int, default=125)
parser.add_argument('--attention_size', type = int, default=250)
parser.add_argument('--decay_period', type = int, default=5)
parser.add_argument('--decay', type = int, default=0.85)

args = parser.parse_args()
torch.manual_seed(args.seed)



def train() :

    train_data, dev_data, word2id, char2id, opts = load_data(vars(args))
    model = FusionNet(opts)

    if args.cuda :
        model = model.cuda()

    dev_batches = get_batches(dev_data, args.batch_size)

    if args.eval :
        print ('load model...')
        model.load_state_dict(torch.load(args.model_dir))
        model.eval()
        model.boundary_evaluate(dev_batches, args.data_path + 'dev_eval.json', answer_file = 'result/' + args.model_dir.split('/')[-1] + '.answers')
        exit()

    train_batches = get_batches(train_data, args.batch_size)
    total_size = len(train_batches)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adamax(parameters, lr = args.lrate)

    lrate = args.lrate
    best_score = 0.0
    for epoch in range(args.epochs) :
        model.train()
        for i, train_batch in enumerate(train_batches) :
           loss = model(train_batch)
           model.zero_grad()
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           if i % 20 == 0:
               print('Epoch = %d, step = %d / %d, loss = %.5f, lrate = %.5f' % (epoch, i, total_size, loss, lrate))

        model.eval()
        exact_match_score, F1 = model.Evaluate(dev_batches, args.data_path + 'dev_eval.json', answer_file = 'result/' + args.model_dir.split('/')[-1] + '.answers')

        if best_score < F1:
            best_score = F1
            print ('saving %s ...' % args.model_dir)
            torch.save(model.state_dict(), args.model_dir)
        if epoch > 0 and epoch % args.decay_period == 0:
            lrate *= args.decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrate


if __name__ == '__main__' :
    train()