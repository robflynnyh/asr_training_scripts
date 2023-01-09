from omegaconf.omegaconf import OmegaConf


def load_audio_model(args, model_class):
    if args.load_pretrained == True:
        model = model_class.from_pretrained(args.pretrained)
        if args.tokenizer != '':
            model.change_vocabulary(new_tokenizer_dir=args.tokenizer, new_tokenizer_type='bpe')
        return model
    else:
        cfg = OmegaConf.load(args.model_config)
        model = model_class(cfg['model'])
        print(f'Loaded model from config file {args.model_config}')
        return model