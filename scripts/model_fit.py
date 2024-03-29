import torch
from train_val_loop import train_loop
from train_val_loop import validation_loop

def fit(model, opt, mode_lst, train_dataloader, val_dataloader, src_maxlen, num_tokens, epochs, val_min, patience, best_chk_path, device, logger):
    validation_loss_min = val_min
    logger.info(f'epoch train_loss val_loss')
    for epoch in range(1,epochs+1):
        train_loss = train_loop(model, opt, mode_lst, train_dataloader, src_maxlen, num_tokens, device)
        validation_loss = validation_loop(model, mode_lst, val_dataloader, src_maxlen, num_tokens, device)
        logger.info(f'{epoch} {train_loss} {validation_loss}')
        if validation_loss < validation_loss_min:
            validation_loss_min = validation_loss
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'val_min': validation_loss_min,
                        'final_steps': opt.get_final_steps(),
                        }, best_chk_path)
            current_patience = 0
        else:
            current_patience += 1
        if current_patience == patience:
            break
    return None
