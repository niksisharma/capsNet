# train_vit_cifar100.py
from timm.loss import SoftTargetCrossEntropy
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import AverageMeter, accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='ViT CIFAR-100 Training')
    parser.add_argument('--batch-size', type=int, default=768, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        print("Using GPU")
        print(torch.cuda.device_count())
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")


    # Data loading and augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size*2, num_workers=4, pin_memory=True)

    # Model configuration
    model = create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        num_classes=100,
        img_size=32,
        patch_size=8,  # (32/8)^2 = 16 patches
        drop_rate=0.1,
    )
    model = model.to(device)
    
    # Training setup
    # criterion = nn.CrossEntropyLoss()
    train_criterion = SoftTargetCrossEntropy().to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.0, prob=0.8, num_classes=100)

    # Metrics tracking
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_acc': [],
        'epoch_sec': [],
        'gpu_mb': []
    }
    best_acc = 0.0

    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        loss_meter = AverageMeter()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = mixup_fn(inputs, targets)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = train_criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), inputs.size(0))

        # Validation
        model.eval()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                top1 = accuracy(outputs, targets, topk=(1,))[0]
                acc_meter.update(top1.item(), inputs.size(0))
                # acc = accuracy(outputs, val_criterion(outputs, targets).argmax(1), topk=(1,))[0]
                # acc_meter.update(acc.item(), inputs.size(0))

        # Update metrics
        epoch_time = time.time() - start_time
        gpu_mem = torch.cuda.max_memory_allocated() // (1024 * 1024)
        
        metrics['epoch'].append(epoch+1)
        metrics['train_loss'].append(round(loss_meter.avg, 4))
        metrics['val_acc'].append(round(acc_meter.avg, 4))
        metrics['epoch_sec'].append(round(epoch_time, 1))
        metrics['gpu_mb'].append(gpu_mem)

        # Save checkpoint
        if acc_meter.avg > best_acc:
            best_acc = acc_meter.avg
            torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')
        
        scheduler.step()
        print(f'Epoch {epoch+1}/{args.epochs} | Loss: {loss_meter.avg:.4f} | Acc: {acc_meter.avg:.2f}%')

    # Save final model and metrics
    torch.save(model.state_dict(), f'{args.output_dir}/final_model.pth')
    with open(f'{args.output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()
