
# from dataset


class RandomAugmentLoader(object):
    def __init__(self, axes, start, stop, step, resize_shape=None):
        self.axes = axes
        self.angles = np.arange(start, stop+1, step, dtype=int)
        self.rotate_expands = (len(self.angles)-1)*len(axes)+1
        self.resize_shape = resize_shape

        self.processed_dir = 'rotate_'+str(list(axes)[-1]+1) +\
                             '_'+str(start)+'_'+str(stop)+'_'+str(step)

        if resize_shape is not None:
            assert isinstance(resize_shape, list) or\
                isinstance(resize_shape, tuple)
            self.scaled_dir = 'scaled'
            self.scaled_dir += '_'+str(resize_shape[0])
            self.scaled_dir += '_'+str(resize_shape[1])
            self.scaled_dir += '_'+str(resize_shape[2])
            self.processed_dir += '_'+self.scaled_dir

    def _conflict_axis(self, axis):
        axes = list(self.axes)
        axes.remove(axis)
        return axes

    def maybe_preprocess(self, root):
        if not root.startswith('/'):
            root = pjoin(os.getcwd(), root)

        filenames = glob(pjoin(root, '*.nii'))
        if not os.path.exists(pjoin(root, self.processed_dir)):
            os.mkdir(pjoin(root, self.processed_dir))
        tasks = []

        for filename in filenames:
            sid = os.path.split(filename)[-1].split('.')[0]
            path = pjoin(root, self.processed_dir, sid)
            if not os.path.exists(path):
                os.mkdir(path)
            if not os.path.exists(pjoin(path, '0.nii')):
                os.symlink(filename, pjoin(path, '0.nii'))
            if not os.path.exists(pjoin(path, 'z0.nii')):
                os.symlink(pjoin(root, sid+'.nii'), pjoin(path, 'z0.nii'))
            i = 1
            for angle in self.angles:
                if angle == 0:
                    continue
                for axis in self.axes:
                    if not os.path.exists(pjoin(path, str(i)+'.nii')):
                        tasks += [dict(original=filename, angle=angle,
                                       axes=self._conflict_axis(axis),
                                       save=pjoin(path, str(i)+'.nii'))]
                    if self.resize_shape is not None and\
                       not os.path.exists(pjoin(path, 'z'+str(i)+'.nii')):
                        p, f = os.path.split(filename)
                        tasks += [dict(original=pjoin(p, self.scaled_dir, f),
                                       angle=angle,
                                       axes=self._conflict_axis(axis),
                                       save=pjoin(path, 'z'+str(i)+'.nii'))]
                    i += 1

        if len(tasks) != 0:
            pool = Pool(20)
            pbar = tqdm(total=len(tasks))
            for _ in pool.imap_unordered(self._process, tasks):
                pbar.update()
            pbar.close()
            pool.close()

    def _process(self, kwarg):
        sample = nib.load(kwarg['original'])
        image = sample.get_data()
        rimage = rotate(image, kwarg['angle'],
                        axes=kwarg['axes'], reshape=False)
        rsample = nib.Nifti1Image(rimage, sample.affine)
        self._preserve_header(sample, rsample)
        nib.save(rsample, kwarg['save'])

    def _preserve_header(self, sample, other):
        for key in sample.header.keys():
            if 'dim' in key or 'srow' in key or 'qoffset' in key:
                continue
            if (sample.header[key] == other.header[key]).all():
                continue
            other.header[key] = sample.header[key]

    def __call__(self, sid, roots, target_roots=None):
        idx = random.choice(range(self.rotate_expands))
        if self.resize_shape is not None:
            z = True if random.random() < 0.5 else False
        else:
            z = False
        images = []

        for root in roots:
            t = str(idx)+'.nii'
            if z:
                t = 'z'+t
            path = pjoin(root, self.processed_dir, sid, t)
            assert os.path.exists(path)
            sample = nib.load(path)
            images += [sample.get_data()]
        target = _extract_descrip(sample)[1]

        if target_roots is not None:
            target = []
            for target_root in target_roots:
                if z:
                    t = 'z'+t
                path = pjoin(target_root, self.processed_dir, t)
                assert os.path.exists(path)
                sample = nib.load(path)
                target += [sample.get_data()]

        return images, target


class ScaledLoader(object):
    def __init__(self, shape, with_original):
        self.shape = np.array(shape)
        self.processed_dir = 'scaled'
        self.processed_dir += '_'+str(shape[0])
        self.processed_dir += '_'+str(shape[1])
        self.processed_dir += '_'+str(shape[2])
        self.with_original = with_original

    def _preserve_header(self, sample, other):
        for key in sample.header.keys():
            if 'dim' in key or 'srow' in key or 'qoffset' in key:
                continue
            if (sample.header[key] == other.header[key]).all():
                continue
            other.header[key] = sample.header[key]

    def maybe_preprocess(self, roots):
        for root in roots:
            pr = pjoin(root, self.processed_dir)
            if not os.path.exists(pr):
                os.mkdir(pr)
            if len(glob(pjoin(pr, '*.nii'))) == len(glob(pjoin(root, '*.nii'))):
                pass
            else:
                self._preprocess(root)

    def _preprocess(self, root):
        filenames = glob(pjoin(root, '*.nii'))
        pr = pjoin(root, self.processed_dir)
        pbar = tqdm(total=len(filenames))
        for filename in filenames:
            sample = nib.load(filename)
            image = sample.get_data()
            zf = self.shape/np.array(image.shape)
            zimage = zoom(image, zf)
            affine = sample.affine
            affine[:3, :3] *= zf
            zsample = nib.Nifti1Image(zimage, affine)
            self._preserve_header(sample, zsample)
            nib.save(zsample, pjoin(pr, os.path.split(filename)[-1]))
            pbar.update()
        pbar.close()

    def __call__(self, sid, roots, target_roots=None):
        samples = [nib.load(pjoin(root, self.processed_dir,
                                  sid+'.nii')) for root in roots]
        if self.with_original:
            samples += [nib.load(pjoin(root, sid+'.nii')) for root in roots]
        target = _extract_descrip(samples[0])[1]
        if target_roots is not None:
            raise NotImplementedError
        images = [_copy_mmap(sample.get_data()) for sample in samples]
        return images, target

# from train


def pretrain():
    timer = SimpleTimer()
    rk = 'k'+str(FLAGS.running_k)
    summary = Summary(port=39199, env=FLAGS.visdom_env)
    # create datasets
    trainset_loader = None
    train_transformer = Compose(
        [
            NineCrop((112, 144, 112)),
            Lambda(lambda pats: torch.stack([
                ToFloatTensor()(p) for p in pats]))
        ])
    trainloader, _, _ = create_dataset_loader(
        trainset_loader, train_transformer,  # train
        None, None,  # valid
    )

    model = create_model(pretrain_model=True)
    model.cuda(FLAGS.devices[0])

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=FLAGS.base_lr,
    #                             nesterov=True, momentum=0.9,
    #                             weight_decay=FLAGS.l2_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.base_lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                               patience=10, verbose=True,
                                               min_lr=1e-5)

    start_epoch = 0
    running_loss = 0
    global_step = start_epoch*len(trainloader)
    pbar = None
    for epoch in range(start_epoch, FLAGS.max_epoch):
        if optimizer.param_groups[0]['lr'] == 1e-5:
            break

        for i, param_group in enumerate(optimizer.param_groups):
            summary.scalar('pretrain_lr', 'k'+str(FLAGS.running_k)+'_'+str(i),
                           epoch, param_group['lr'],
                           ytickmin=0, ytickmax=FLAGS.base_lr)

        if pbar is None:
            pbar = tqdm(total=len(trainloader),
                        desc='Epoch {:>4}'.format(epoch))

        images = None
        target = None
        outputs = None
        for images, target in trainloader:
            loss = None
            npatches = 1

            if isinstance(target, torch.LongTensor):
                target = images.clone()
            if len(images.shape) == 6:
                b, npatches, c, x, y, z = images.shape
                images = images.view(-1, c, x, y, z)
                if isinstance(target, list) or isinstance(target, tuple):
                    target = torch.cat([t.view(-1, c, x, y, z) for t in target],
                                       dim=1)
                else:
                    target = target.view(-1, c, x, y, z)

            # TODO multi-modal
            images = Variable(images.cuda(FLAGS.devices[0], async=True))
            target = Variable(target.cuda(FLAGS.devices[0], async=True))

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if global_step > start_epoch*len(trainloader) and\
               global_step % FLAGS.graph_term == 0:
                print_loss = running_loss/FLAGS.graph_term
                summary.scalar(
                    rk+'_pretrain_loss', 'train',
                    global_step/len(trainloader), print_loss,
                    ytickmin=0, ytickmax=0.03)
                running_loss = 0

            global_step += 1
            pbar.update()
        pbar.close()
        pbar = None

        summary.image3d(rk+'_input', images)
        if FLAGS.multi_task != 1:
            for c in range(outputs.size(1)):
                summary.image3d(rk+'_output_'+str(c),
                                outputs[:, c, ...].unsqueeze(1))
                summary.image3d(rk+'_target_'+str(c),
                                target[:, c, ...].unsqueeze(1))
        else:
            summary.image3d(rk+'_target', target)
            summary.image3d(rk+'_output', outputs)
        scheduler.step(print_loss)
        if epoch % 10 == 0:
            save_checkpoint(dict(epoch=None,
                                 state_dict=model.module.state_dict(),
                                 best_score=None,
                                 optimizer=None),
                            FLAGS.checkpoint_root, FLAGS.running_k,
                            FLAGS.model+'_cae', is_best=False)

    save_checkpoint(dict(epoch=None,
                         state_dict=model.module.state_dict(),
                         best_score=None,
                         optimizer=None),
                    FLAGS.checkpoint_root, FLAGS.running_k,
                    FLAGS.model+'_cae', is_best=False)
