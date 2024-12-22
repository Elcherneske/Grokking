import argparse
from torch.utils.data import DataLoader
from BinaryDataset import BinaryDataset
from Model import MLP
from Trainer import ModelTrainer
from Loss import CELoss

def init_arg_parser():
    parser = argparse.ArgumentParser()
    #database parameters
    parser.add_argument('--data_file_path', default='./dataset/data.pkl')
    parser.add_argument('--dataset', default='binary')
    #model parameters
    parser.add_argument('--mlp', default=[128, 512, 97])
    parser.add_argument('--batch_size', default=64, type=int)
    #train parameters
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--mode", choices=["train_eva"], default="train_eva")
    #loss parameters
    parser.add_argument('--loss_function', default='CrossEntropyLoss', type=str)
    return parser


def main():
    parser = init_arg_parser()
    args = parser.parse_args()
    device = args.device

    if args.dataset == 'binary':
        train_dataset = BinaryDataset(filename=args.data_file_path, is_train=True, device=device)
        test_data =  BinaryDataset(filename=args.data_file_path, is_train=False, device=device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: filename: {args.data_file_path}")

    model = MLP(mlp=[128, 512, 97]).to(device)
    print(f"parameter number: {sum(p.numel() for p in model.parameters())}")

    trainer = ModelTrainer(model=model, optimizer=args.optimizer, lr=args.lr)
    if args.loss_function == "CrossEntropyLoss":
        loss_fn = CELoss()

    print(f"Trainner Params: epochs: {args.epoch}, batch: {args.batch_size}, lr: {args.lr}, device: {args.device}, mode:{args.mode}")
    if args.mode == "train_eva":
        trainer.train_and_validation(criterion=loss_fn, train_loader=train_loader, val_loader=test_loader, num_epochs=args.epoch)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
