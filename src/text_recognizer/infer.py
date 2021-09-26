import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from .modules.model_utils import CTCLabelConverter, AttnLabelConverter
from .modules.dataset import RawDataset, AlignCollate
from .modules.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_preparation(opt, list_data):
    AlignCollate_obj = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    dataset = RawDataset(list_data=list_data, opt=opt)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=int(opt.workers),
                                              collate_fn=AlignCollate_obj, pin_memory=True)
    return data_loader


def inference(opt, model, converter, data_loader):
    output_pred, output_conf_score = [], []
    model.eval()
    with torch.no_grad():
        for image_tensors, _ in data_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor(
                [opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(
                batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            preds = model(image, text_for_pred, is_train=False)
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                # prune after "end of sentence" token ([s])
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]
                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                output_pred.append(pred)
                output_conf_score.append(confidence_score)
    return output_pred, output_conf_score
