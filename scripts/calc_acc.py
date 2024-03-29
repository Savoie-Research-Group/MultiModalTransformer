#Calculating top-n accuracy and return the number

from rdkit import Chem
from beam_search import beam_search

def bs_accuracy(batch, model, src_maxlen, tgt_maxlen, idx_to_ch_dict, target_start_token_idx, target_end_token_idx):
    count_top1 = 0
    count_top5 = 0
    beam_size = 5
    source, target = batch
    bs = source.size(dim=0)
    preds = beam_search(model, source, beam_size, src_maxlen, tgt_maxlen, target_start_token_idx, target_end_token_idx)
    for i in range(bs):
        target_text = "".join([idx_to_ch_dict[int(_)] for _ in target[i, :]])
        target_text = target_text.replace("PAD_WORD","")
        target_text = target_text.replace("<","")
        target_text = target_text.replace("$","")
        preds[i].sort(key = lambda x:x[1])
        pred_dict = {}
        for j in range(beam_size):
            pred_dict["prediction_{}".format(j)] = ""
            for idx in preds[i][-1-j][0]:
                pred_dict["prediction_{}".format(j)] += idx_to_ch_dict[idx]
                if idx == target_end_token_idx:
                    break
            pred_dict["prediction_{}".format(j)] = pred_dict["prediction_{}".format(j)].replace("<","")
            pred_dict["prediction_{}".format(j)] = pred_dict["prediction_{}".format(j)].replace("$","")
            try:
                pred_dict["prediction_{}".format(j)] = Chem.CanonSmiles(pred_dict["prediction_{}".format(j)])
            except:
                pred_dict["prediction_{}".format(j)] = "XXX"

        prediction_top_1 = pred_dict["prediction_0"]
        prediction_top_5 = []
        for k in range(5):
            prediction_top_5 += [pred_dict["prediction_{}".format(k)]]
        if prediction_top_1 == target_text:
            count_top1 = count_top1 + 1
        if target_text in prediction_top_5:
            count_top5 = count_top5 + 1

#Uncomment this section if you need to print out the smiles
        with open('test-multi1.txt','a') as f:
            for mol in prediction_top_5:
                f.write(mol)
                f.write(';')
            f.write(target_text)
            f.write('\n')
    return count_top1, count_top5

