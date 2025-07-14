import string
import argparse
import time  # Added for infer_time calculation
import sys  # Added for platform check

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from nltk.metrics.distance import edit_distance
from dataset import hierarchical_dataset, AlignCollate  # Changed from RawDataset
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_normalized_ed(pred, gt):
    """
    Calculates Normalized Edit Distance (NED) following the logic from train.py,
    using the edit_distance function from utils.
    """
    distance = edit_distance(pred, gt)  # Use the imported edit_distance

    if len(gt) == 0 or len(pred) == 0:
        return 0.0
    elif len(gt) > len(pred):
        return 1 - (distance / len(gt))
    else:
        return 1 - (distance / len(pred))


def evaluate(opt):
    """model configuration"""
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print(
        "model input parameters",
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction,
    )
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print("loading pretrained model from %s" % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data for evaluation
    AlignCollate_eval = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
    )
    # Using hierarchical_dataset for evaluation, similar to train.py's validation
    eval_dataset, eval_dataset_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=int(opt.workers),  # Use the potentially modified opt.workers
        collate_fn=AlignCollate_eval,
        pin_memory=True,
    )

    print(eval_dataset_log)  # Print dataset log

    # Evaluation loop
    model.eval()
    correct_prediction = 0
    total_data = 0
    norm_ED_score_avg = Averager()  # Averager for Normalized Edit Distance
    infer_time_total = 0

    log = open(f"./log_demo_result.txt", "a", encoding="utf-8")
    dashed_line = "-" * 80
    head = f'{"Ground Truth":25s}\t{"Prediction":25s}\tConfidence Score\tCorrect\tNormalized ED'

    print(f"{dashed_line}\n{head}\n{dashed_line}")
    log.write(f"{dashed_line}\n{head}\n{dashed_line}\n")

    with torch.no_grad():
        for i, (image_tensors, labels) in enumerate(eval_loader):  # Now we get labels
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(
                device
            )
            text_for_pred = (
                torch.LongTensor(batch_size, opt.batch_max_length + 1)
                .fill_(0)
                .to(device)
            )

            start_time = time.time()
            if "CTC" in opt.Prediction:
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
            infer_time_total += time.time() - start_time

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
                if "Attn" in opt.Prediction:
                    pred_EOS = pred.find("[s]")
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # Calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                # Compare prediction with ground truth for accuracy
                is_correct = pred == gt
                if is_correct:
                    correct_prediction += 1

                # Calculate Normalized Edit Distance
                norm_ed = get_normalized_ed(pred, gt)
                # Convert norm_ed to a tensor before adding to Averager
                norm_ED_score_avg.add(
                    torch.tensor(norm_ed, dtype=torch.float32).to(device)
                )

                print(
                    f"{gt:25s}\t{pred:25s}\t{confidence_score:0.4f}\t{str(is_correct):5s}\t{norm_ed:0.4f}"
                )
                log.write(
                    f"{gt:25s}\t{pred:25s}\t{confidence_score:0.4f}\t{str(is_correct):5s}\t{norm_ed:0.4f}\n"
                )

            total_data += batch_size

    accuracy = correct_prediction / total_data * 100
    avg_norm_ED = norm_ED_score_avg.val()

    print(f"{dashed_line}")
    print(f"Evaluation finished. Total samples: {total_data}")
    print(f"Accuracy: {accuracy:0.2f}%")
    print(f"Normalized Edit Distance (NED): {avg_norm_ED:0.4f}")
    print(f"Inference time: {infer_time_total:0.4f}s")

    log.write(f"{dashed_line}\n")
    log.write(f"Evaluation finished. Total samples: {total_data}\n")
    log.write(f"Accuracy: {accuracy:0.2f}%\n")
    log.write(f"Normalized Edit Distance (NED): {avg_norm_ED:0.4f}\n")
    log.write(f"Inference time: {infer_time_total:0.4f}s\n")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Changed from --image_folder to --eval_data for dataset with labels
    parser.add_argument(
        "--eval_data",
        required=True,
        help="path to evaluation dataset which contains text images and labels",
    )
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=192, help="input batch size")
    parser.add_argument(
        "--saved_model", required=True, help="path to saved_model for evaluation"
    )
    """ Data processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=64, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument("--rgb", action="store_true", help="use rgb input")
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyz",
        help="character label",
    )
    parser.add_argument(
        "--sensitive", action="store_true", help="for sensitive character mode"
    )
    parser.add_argument(
        "--PAD",
        action="store_true",
        help="whether to keep ratio then pad for image resize",
    )
    # Added the missing argument for data filtering
    parser.add_argument(
        "--data_filtering_off", action="store_true", help="for data_filtering_off mode"
    )
    """ Model Architecture """
    parser.add_argument(
        "--Transformation",
        type=str,
        required=True,
        help="Transformation stage. None|TPS",
    )
    parser.add_argument(
        "--FeatureExtraction",
        type=str,
        required=True,
        help="FeatureExtraction stage. VGG|RCNN|ResNet",
    )
    parser.add_argument(
        "--SequenceModeling",
        type=str,
        required=True,
        help="SequenceModeling stage. None|BiLSTM",
    )
    parser.add_argument(
        "--Prediction", type=str, required=True, help="Prediction stage. CTC|Attn"
    )
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=1,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=256,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )

    opt = parser.parse_args()
    opt.character = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ªÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿĀāĂăĄąĆćČčĎčĐđĒēĖėĘęĚěĞğĨĩĪīĮįİıĶķĹĺĻļĽľŁłŃńŅņŇňŒœŔŕŘřŚśŞşŠšŤťŨũŪūŮůŲųŸŹźŻżŽžƏƠơƯưȘșȚțə̇ḌḍḶḷḀṁṂṃṄṅṆṇṬṭẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ€"

    """ vocab / character number configuration """
    # if opt.sensitive:
    #     opt.character = string.printable[:-6] # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # Set num_workers to 0 for Windows to avoid pickling issues with DataLoader
    if sys.platform == "win32":
        print(
            "Detected Windows OS. Setting num_workers to 0 to avoid multiprocessing issues."
        )
        opt.workers = 0

    evaluate(opt)  # Call the renamed function
