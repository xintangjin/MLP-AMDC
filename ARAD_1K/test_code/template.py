def set_template(args):
    # Set the templates here
    if args.method.find('AMDC') >= 0:
        args.use_adaptive_mask = True
        args.input_setting = 'AMDC'
        args.loss = True

    if args.method.find('OLU') >= 0:
        args.input_setting = 'Y'
        args.loss = True

    if args.method.find('mst') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Phi'

    if args.method.find('gap_net') >= 0 or args.method.find('admm_net') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'
        args.milestones = range(30,args.max_epoch,30)
        args.gamma = 0.9
        args.learning_rate = 1e-3

    if args.method.find('dauhst') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300

    if args.method.find('tsa_net') >= 0:
        args.input_setting = 'HM'
        args.input_mask = None

    if args.method.find('hdnet') >= 0:
        args.input_setting = 'H'
        args.input_mask = None


    if args.method.find('mst_plus_plus') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
    
    if args.method.find('cst') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 500

    if args.method.find('dnu') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'
        args.batch_size = 2
        args.max_epoch = 150
        args.milestones = range(10,args.max_epoch,10)
        args.gamma = 0.9
        args.learning_rate = 4e-4

    if args.method.find('lambda_net') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        args.learning_rate = 1.5e-4