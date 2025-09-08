import torch
from tqdm import tqdm

def multimodaltrain(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training Multimodal", leave=False)
    
    for X_batch, X_tab_batch, y_batch in progress_bar:
        X_batch, X_tab_batch, y_batch = X_batch.to(device), X_tab_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs, latents = model(X_batch, X_tab_batch, return_latents=True)

        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
        progress_bar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{correct / total:.4f}"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Training function
# def multimodaltrain(model, loader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     for X_batch, X_tab_batch, y_batch in loader:
#         X_batch, X_tab_batch, y_batch = X_batch.to(device), X_tab_batch.to(device), y_batch.to(device)
        
#         optimizer.zero_grad()
#         # outputs = model(X_batch, X_tab_batch)

#         outputs, latents = model(X_batch, X_tab_batch, return_latents=True)

#         loss = criterion(outputs, y_batch)
#         loss.backward()
        
#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()

#         running_loss += loss.item() * X_batch.size(0)
#         _, preds = torch.max(outputs, 1)
#         correct += (preds == y_batch).sum().item()
#         total += y_batch.size(0)

#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#     return epoch_loss, epoch_acc

def multimodalevaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating Multimodal", leave=False)
        for X_batch, X_tab_batch, y_batch in progress_bar:
            X_batch, X_tab_batch, y_batch = X_batch.to(device), X_tab_batch.to(device), y_batch.to(device)
            outputs, latents = model(X_batch, X_tab_batch, return_latents=True)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
            progress_bar.set_postfix({
                "loss": f"{running_loss / total:.4f}",
                "acc": f"{correct / total:.4f}"
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# def multimodalevaluate(model, loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for X_batch, X_tab_batch, y_batch in loader:
#             X_batch, X_tab_batch, y_batch = X_batch.to(device), X_tab_batch.to(device), y_batch.to(device)
#             outputs, latents  = model(X_batch, X_tab_batch, return_latents=True)
#             loss = criterion(outputs, y_batch)

#             running_loss += loss.item() * X_batch.size(0)
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == y_batch).sum().item()
#             total += y_batch.size(0)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y_batch.cpu().numpy())

#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#     return epoch_loss, epoch_acc, all_preds, all_labels


# Training function with tqdm progress bar
def tabtrain(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for X_batch, y_batch in progress_bar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        # Optionally update progress bar postfix with current metrics
        progress_bar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{correct / total:.4f}"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# def tabtrain(model, loader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     for X_batch, y_batch in loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(X_batch)

#         loss = criterion(outputs, y_batch)
#         loss.backward()
        
#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()

#         running_loss += loss.item() * X_batch.size(0)
#         _, preds = torch.max(outputs, 1)
#         correct += (preds == y_batch).sum().item()
#         total += y_batch.size(0)

#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#     return epoch_loss, epoch_acc


def tabevaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            progress_bar.set_postfix({
                "loss": f"{running_loss / total:.4f}",
                "acc": f"{correct / total:.4f}"
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# Evaluation function
# def tabevaluate(model, loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for X_batch, y_batch in loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             running_loss += loss.item() * X_batch.size(0)
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == y_batch).sum().item()
#             total += y_batch.size(0)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y_batch.cpu().numpy())

#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#     return epoch_loss, epoch_acc, all_preds, all_labels



def cnntrain(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def cnnevaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc




class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss  # we minimize val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0



def run_training(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, patience=5):
    early_stopper = EarlyStopping(patience=patience, verbose=True)
    model.to(device)

    for epoch in range(num_epochs):
        train_loss, train_acc = cnntrain(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = cnnevaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model
    model.load_state_dict(early_stopper.best_model_state)
    return model


if __name__=="__main__":
    pass