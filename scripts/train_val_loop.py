import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

def train_loop(model, opt, mode_lst, dataloader, src_maxlen, num_tokens, device):
    model.train()
    gc.collect()
    total_loss = 0
    for batch in dataloader:
        X, y = batch[0], batch[1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)
        x_src = X[:,0:src_maxlen]
        if mode_lst[0] == 1:
            x_ms = X[:,src_maxlen:src_maxlen+999]
        else:
            x_ms = None
        if mode_lst[1] == 1:
            x_ir = X[:,src_maxlen+999:src_maxlen+1899]
        else:
            x_ir = None
        if mode_lst[2] == 1:
            x_nmr = X[:,src_maxlen+1899:]
        else:
            x_nmr = None
        # shift the tgt by one so with the start token we predict the token at pos 1
        y_input = y[:,:-1]
        y_expected = y[:,1:]
        # Create a sample weight mask to mask out loss value calculated on padding
        sample_weight = torch.logical_not(torch.eq(y_expected,0))
        sample_weight = sample_weight.long()
        y_expected = F.one_hot(y_expected.long(),num_classes = num_tokens)
        pred = model(x_src, x_ms, x_ir, x_nmr, y_input)
        # permute pred and y_expect so num_classes is the second dimension
        pred = torch.permute(pred,(0,2,1))
        y_expected = torch.permute(y_expected,(0,2,1))
        loss = nn.CrossEntropyLoss(reduction="none")(pred, y_expected.float())
        loss = (loss * sample_weight / sample_weight.sum()).sum()
        opt.zero_grad()
        loss.backward()
        opt.step_and_update_lr()
        total_loss += loss.detach().item()
    return total_loss / len(dataloader)

def validation_loop(model, mode_lst, dataloader, src_maxlen, num_tokens, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[0], batch[1]
            X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)
            x_src = X[:,0:src_maxlen]
            if mode_lst[0] == 1:
                x_ms = X[:,src_maxlen:src_maxlen+999]
            else:
                x_ms = None
            if mode_lst[1] == 1:
                x_ir = X[:,src_maxlen+999:src_maxlen+1899]
            else:
                x_ir = None
            if mode_lst[2] == 1:
                x_nmr = X[:, src_maxlen+1899:]
            else:
                x_nmr = None
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            sample_weight = torch.logical_not(torch.eq(y_expected,0))
            sample_weight = sample_weight.long()
            y_expected = F.one_hot(y_expected.long(),num_classes = num_tokens)
            pred = model(x_src, x_ms, x_ir, x_nmr, y_input)
            pred = torch.permute(pred,(0,2,1))
            y_expected = torch.permute(y_expected,(0,2,1))
            loss = nn.CrossEntropyLoss(reduction="none")(pred, y_expected.float())
            loss = (loss * sample_weight / sample_weight.sum()).sum()
            total_loss += loss.detach().item()
    return total_loss / len(dataloader)
