import torch
import torch.nn as nn
import torch.nn.functional as F

def Perplexity(output, target):
    with torch.no_grad():
        ce = F.cross_entropy(output, target)
        perplexity = torch.exp(ce)
    return perplexity


def evaluate(model, test_loader, args, device):
    
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    loss, total, correct_multi= 0.0, 0.0, 0.0
    accuracy_single_list = list()
    
    for i in range(args.num_branch):
        accuracy_single_list.append(0)

    with torch.no_grad():
        if args.dataset != 'wikitext-2':
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                output_list, _ = model(images)

                ensemble_output = torch.stack(output_list, dim=2)
                ensemble_output = torch.sum(ensemble_output, dim=2) / len(output_list)
                
                _, pred_labels_multi = torch.max(ensemble_output, 1)
                pred_labels_multi = pred_labels_multi.view(-1)
                correct_multi += torch.sum(torch.eq(pred_labels_multi, labels)).item()

                for i, single in enumerate(output_list):  
                    _, pred_labels_single = torch.max(single, 1)
                    pred_labels_single = pred_labels_single.view(-1)
                    accuracy_single_list[i] += torch.sum(torch.eq(pred_labels_single, labels)).item()
                    
                total += len(labels)
        else:
            for _, input in enumerate(test_loader):
                input = input.to(device)

                mask = torch.rand(input.shape) < 0.15
                mask_change = mask & (torch.rand(input.shape) < 0.9)
                mask_random = mask_change & (torch.rand(input.shape) < 1/9)

                output_list = model(input, mask_change, mask_random)

                ensemble_output = torch.stack(output_list, dim=2)
                ensemble_output = torch.sum(ensemble_output, dim=2) / len(output_list)
                correct_multi += Perplexity(ensemble_output[mask], input[mask]).item()

                for i, single in enumerate(output_list):
                    accuracy_single_list[i] += Perplexity(output_list[i][mask], input[mask]).item()
                
                total += len(input)

        accuracy_multi = correct_multi/total

        for i in range(len(accuracy_single_list)):
            accuracy_single_list[i] /= total
        
    model.to(torch.device('cpu'))
    
    return accuracy_multi, accuracy_single_list, loss
#gunhot
def evaluate_simple(model, train_loader, test_loader, args, device):
    # model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss, total, correct = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # logits = outputs.logits
            # loss = outputs.loss
            test_loss += loss.mean().item()

            _, pred_labels = torch.max(logits, dim=-1)

            # ignore padding (-100)
            mask = labels != -100
            correct += torch.sum((pred_labels == labels) & mask).item()
            total += mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    train_loss = 0.0

    model.to(torch.device('cpu'))
    return accuracy, train_loss, test_loss
#gunhot

# def evaluate_simple(model, train_loader, test_loader, args, device):
    
#     model = nn.DataParallel(model)
#     model.to(device)
#     model.eval()
#     criterion = nn.CrossEntropyLoss()

#     test_loss, total, correct = 0.0, 0.0, 0.0
    
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(test_loader):
#             images, labels = images.to(device), labels.to(device)
#             output = model(images)

#             test_loss += criterion(output, labels).to(torch.device('cpu'))
#             _, pred_labels = torch.max(output, 1)
#             pred_labels = pred_labels.view(-1)
#             correct += torch.sum(torch.eq(pred_labels, labels)).item()

#             total += len(labels)

#         accuracy = correct/total

#     train_loss = 0.0
    
#     # with torch.no_grad():
#     #     for batch_idx, (images, labels) in enumerate(train_loader):
#     #         images, labels = images.to(device), labels.to(device)
#     #         output = model(images)

#     #         train_loss += criterion(output, labels).to(torch.device('cpu'))
            
#     model.to(torch.device('cpu'))
    
#     return accuracy, train_loss, test_loss
    
# if __name__ == "__main__":
#     print("Execute models.py")