from omegaconf.omegaconf import OmegaConf


def get_model_class(args, classname=None):
    from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
    from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
    from nemo.collections.asr.models.scctc_bpe_models import EncDecSCCTCModelBPE
    from nemo.collections.asr.models.s4_scctc_bpe_models import s4EncDecSCCTCModelBPE
    assert classname is not None or 'model_class' in args.__dict__, 'Must specify model class name either in args as model_class or as classname'
    if classname is None:
        classname = args.model_class
    classes = {
        'EncDecCTCModelBPE': EncDecCTCModelBPE,
        'EncDecRNNTBPEModel': EncDecRNNTBPEModel,
        'EncDecSCCTCModelBPE': EncDecSCCTCModelBPE,
        's4EncDecSCCTCModelBPE': s4EncDecSCCTCModelBPE,
    }
    assert classname in classes, f'Class {classname} not found in {classes.keys()}'
    return classes[classname]

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
