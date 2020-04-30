from features.build_features import letterToVector, nameToTensor
from torch.autograd import Variable
import torch



def predict(model_path, name, categories):
    rnn = torch.load(model_path)
    return predict_line(rnn, name, categories)


def evaluate_line(model, line_tensor):
    hidden = torch.zeros(1, model.hidden_size)
    
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    
    return output

def predict_line(model, line, categories, n_predictions=3):
    output = evaluate_line(model, Variable(nameToTensor(line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, categories[category_index]))
        predictions.append([value, categories[category_index]])

    return predictions