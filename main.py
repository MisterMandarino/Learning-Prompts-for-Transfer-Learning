from utils.datasets import *
from models.coop import *
from models.cocoop import *
from models.MaPLe import *
from utils.train import train, save_model
from utils.CLIP_zs import *
from utils.plot import *

import os
HOME = os.getcwd()
WEIGHTS_PATH = os.path.join(HOME, 'weights')

## Load CLIP
clip_model, preprocess = clip.load("ViT-B/32")
#clip_model, preprocess = clip.load("RN50")
clip_model.to(device)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASETS = ['cifar10', 'cifar100', 'food101', 'oxford pets', 'mnist']

# Set number of epochs
NUM_EPOCHS = 5

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

## Get Datasets ##
cifar10_train_loader, cifar10_test_loader, cifar10_mapper = get_Cifar_10_dataset(32, install=True)
cifar100_train_loader, cifar100_test_loader, cifar100_mapper = get_Cifar_100_dataset(32, install=True)
food101_train_loader, food101_test_loader, food101_mapper = get_food_101_dataset(32, install=True)
pets_train_loader, pets_test_loader, pets_mapper = get_oxford_pet_dataset(32, install=True)
mnist_train_loader, mnist_test_loader, mnist_mapper = get_MNIST_dataset(32, install=True)

classes_dict = {}
results_zero_shot = {}
for dataset in DATASETS:
    print(f'Zero-shot CLIP with {dataset} dataset')

    if dataset == 'cifar10':
        classes = list(cifar10_mapper.keys())
    elif dataset == 'cifar100':
        classes = [cls.replace('_',' ') for cls in cifar100_mapper.keys()]
    elif dataset == 'food101':
        classes = [cls.replace('_',' ') for cls in food101_mapper.keys()]
    elif dataset == 'oxford pets':
        classes = pets_mapper.values()
    elif dataset == 'mnist':
        classes = list(mnist_mapper.keys())

    classes_dict[dataset] = classes
    
    # instantiate the model
    model = CLIP_ZeroShot(classes, clip_model)
    ## Turning off gradients
    for param in model.parameters():
        param.requires_grad_(False)
    model = model.to(device)

    # Evaluate the model
    if dataset == 'cifar10':
        result = test_clip(model=model, data_loader=cifar10_test_loader)
    elif dataset == 'cifar100':
        result = test_clip(model=model, data_loader=cifar100_test_loader)
    elif dataset == 'food101':
        result = test_clip(model=model, data_loader=food101_test_loader)
    elif dataset == 'oxford pets':
        result = test_clip(model=model, data_loader=pets_test_loader)
    elif dataset == 'mnist':
        result = test_clip(model=model, data_loader=mnist_test_loader)

    results_zero_shot[dataset] = float(result.cpu())

    print(f"Test Accuracy: {float(result.cpu()):.3}")

## Train CoOp Model ##
results_coop = {}
for dataset in DATASETS:
    print(f'training CoOp on {dataset}...')

    # instantiate the model
    model = CoOp(classes_dict[dataset], clip_model)

    ## Turning off gradients in both the image and the text encoder
    for name, param in model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)

    model = model.to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()

    # NOTE: only give prompt_learner to the optimizer
    optimizer = torch.optim.SGD(params=model.prompt_learner.parameters(), lr=0.002)

    # Train model
    if dataset == 'oxford pets':
        results = train(model=model, train_loader=pets_train_loader, test_loader=pets_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'food101':
        results = train(model=model, train_loader=food101_train_loader, test_loader=food101_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'cifar100':
        results = train(model=model, train_loader=cifar100_train_loader, test_loader=cifar100_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'cifar10':
        results = train(model=model, train_loader=cifar10_train_loader, test_loader=cifar10_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'mnist':
        results = train(model=model, train_loader=mnist_train_loader, test_loader=mnist_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)

    results_coop[dataset] = results

    ## Save the model
    model_name = f'CoOp_{dataset}.pt'
    save_model(model=model, target_dir=WEIGHTS_PATH, model_name=model_name)

plot_coop(NUM_EPOCHS, DATASETS, results_coop, results_zero_shot)

## CoOp Hyperparameters ##
class_token_positions = ['end', 'middle']
class_specific_context = [True, False]
context_tokens = [4, 8, 16]
init_words = 'a photo of a'

for dataset in DATASETS:

    plt.figure(figsize=(8,8))
    plt.scatter(x=0, y=results_zero_shot[dataset], marker='x', c='red', label='zero-shot CLIP')

    for class_t in class_token_positions:
        for class_s in class_specific_context:
            print(f'training on {dataset}... (class position = [{class_t}] / class-specific = [{class_s}])')

            # instantiate the model
            model = CoOp(classes_dict[dataset], clip_model, n_context=8, init_words='', class_token_pos=class_t, csc=class_s)

            ## Turning off gradients in both the image and the text encoder
            for name, param in model.named_parameters():
                        if "prompt_learner" not in name:
                            param.requires_grad_(False)

            model = model.to(device)

            # Setup loss function and optimizer
            loss_fn = nn.CrossEntropyLoss()

            # NOTE: only give prompt_learner to the optimizer
            optimizer = torch.optim.SGD(params=model.prompt_learner.parameters(), lr=0.002)

            # Train model
            if dataset == 'oxford pets':
                model_results = train(model=model, train_loader=pets_train_loader, test_loader=pets_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
            elif dataset == 'food101':
                model_results = train(model=model, train_loader=food101_train_loader, test_loader=food101_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
            elif dataset == 'cifar10':
                model_results = train(model=model, train_loader=cifar10_train_loader, test_loader=cifar10_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
            elif dataset == 'cifar100':
                model_results = train(model=model, train_loader=cifar100_train_loader, test_loader=cifar100_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)

            plt.plot(range(1, NUM_EPOCHS+1), model_results['test_acc'], label=f'pos={class_t}, csc={class_s}')

    ## Plot the comparison between hyperparameters
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(dataset)
    plt.legend()
    plt.show()

## Draw UMAP of the DATASETS ##
for dataset in DATASETS:

    # Get image and label from custom DataLoader 
    if dataset == 'oxford pets':
        img, label = next(iter(pets_train_loader))
    elif dataset == 'food101':
        img, label = next(iter(food101_train_loader))
    elif dataset == 'cifar100':
        img, label = next(iter(cifar100_train_loader))
    elif dataset == 'cifar10':
        img, label = next(iter(cifar10_train_loader))
    elif dataset == 'mnist':
        img, label = next(iter(mnist_train_loader))
 
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes_dict[dataset]]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(img)
        text_features = clip_model.encode_text(text_inputs)

    data = image_features.numpy()
    prompts = text_features.numpy()
    target = label.numpy()

    draw_umap(data, target, classes_dict[dataset], n_components=3, title=f'{dataset} (CLIP Visual Embedding)')

## UMAP CLIP Embedding space with Learned Prompts from pretrained model
for dataset in DATASETS:
    ## load pre-trained coop
    model_name = f'CoOp_{dataset}.pt'
    model = CoOp(classes_dict[dataset], clip_model)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH, model_name)))
    model = model.to(device)
    model.eval()

    # Get image and label from custom DataLoader
    if dataset == 'oxford pets':
        img, label = next(iter(pets_train_loader))
    elif dataset == 'food101':
        img, label = next(iter(food101_train_loader))
    elif dataset == 'cifar100':
        img, label = next(iter(cifar100_train_loader))
    elif dataset == 'cifar10':
        img, label = next(iter(cifar10_train_loader))
    elif dataset == 'mnist':
        img, label = next(iter(mnist_train_loader))

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes_dict[dataset]]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    img = img.to(device)
    image_features, text_features_learned, probits = model(img)

    image_features = image_features.cpu().detach().numpy()
    text_features = text_features.cpu().detach().numpy()
    text_features_learned = text_features_learned.cpu().detach().numpy()
    target = label.numpy()

    if dataset == 'oxford pets':
        draw_coop_umap(image_features, text_features, text_features_learned, target, classes_dict[dataset], pets_mapper, title=f'{dataset} (CLIP Visual Embedding)')
    elif dataset == 'food101':
        draw_coop_umap(image_features, text_features, text_features_learned, target, classes_dict[dataset], food101_mapper, title=f'{dataset} (CLIP Visual Embedding)')
    elif dataset == 'cifar100':
        draw_coop_umap(image_features, text_features, text_features_learned, target, classes_dict[dataset], cifar100_mapper, title=f'{dataset} (CLIP Visual Embedding)')
    elif dataset == 'cifar10':
        draw_coop_umap(image_features, text_features, text_features_learned, target, classes_dict[dataset], cifar10_mapper, title=f'{dataset} (CLIP Visual Embedding)')
    elif dataset == 'mnist':
        draw_coop_umap(image_features, text_features, text_features_learned, target, classes_dict[dataset], mnist_mapper, title=f'{dataset} (CLIP Visual Embedding)')

## TRAIN COCOOP MODEL ##
results_cocoop = {}
for dataset in DATASETS:
    print(f'training CoOp on {dataset}...')

    # instantiate the model
    model = CoCoOp(classes_dict[dataset], clip_model)
    ## Turning off gradients in both the image and the text encoder
    for name, param in model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
    model = model.to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()

    # NOTE: only give prompt_learner to the optimizer
    optimizer = torch.optim.SGD(params=model.prompt_learner.parameters(), lr=0.002)

    # Train model
    if dataset == 'oxford pets':
        results = train(model=model, train_loader=pets_train_loader, test_loader=pets_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'food101':
        results = train(model=model, train_loader=food101_train_loader, test_loader=food101_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'cifar100':
        results = train(model=model, train_loader=cifar100_train_loader, test_loader=cifar100_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'cifar10':
        results = train(model=model, train_loader=cifar10_train_loader, test_loader=cifar10_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'mnist':
        results = train(model=model, train_loader=mnist_train_loader, test_loader=mnist_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)

    results_cocoop[dataset] = results

    ## Save the model
    model_name = f'CoCoOp_{dataset}.pt'
    save_model(model=model, target_dir=WEIGHTS_PATH, model_name=model_name)


## TRAIN MAPLE MODEL ##
for dataset in DATASETS:
    print(f'training CoOp on {dataset}...')

    # instantiate the model
    model = MaPLe(classes_dict[dataset], clip_model)
    ## Turning off gradients in both the image and the text encoder
    for name, param in model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
    model = model.to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()

    # NOTE: only give prompt_learner to the optimizer
    optimizer = torch.optim.SGD(params=model.prompt_learner.parameters(), lr=0.002)

    ## Train model
    if dataset == 'oxford pets':
        results = train(model=model, train_loader=pets_train_loader, test_loader=pets_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'food101':
        results = train(model=model, train_loader=food101_train_loader, test_loader=food101_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'cifar100':
        results = train(model=model, train_loader=cifar100_train_loader, test_loader=cifar100_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'cifar10':
        results = train(model=model, train_loader=cifar10_train_loader, test_loader=cifar10_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    elif dataset == 'mnist':
        results = train(model=model, train_loader=mnist_train_loader, test_loader=mnist_test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)

    ## Save the model
    model_name = f'MaPLe_{dataset}.pt'
    save_model(model=model, target_dir=WEIGHTS_PATH, model_name=model_name)