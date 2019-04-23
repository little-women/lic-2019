# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-23 20:59:04
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-23 21:08:22


class Mem2Seq(nn.Module):

    def __init__(self, hidden_size, max_len, max_r, lang, path, task, lr, n_layers, dropout, unk_mask):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len  # max input
        self.max_r = max_r  # max responce len
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.unk_mask = unk_mask

        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th')
                self.decoder = torch.load(str(path) + '/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(
                    str(path) + '/enc.th', lambda storage, loc: storage)
                self.decoder = torch.load(
                    str(path) + '/dec.th', lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(
                lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
            self.decoder = DecoderMemNN(
                lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_vac = self.loss_vac / self.print_every
        self.print_every += 1
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg, print_loss_vac, print_loss_ptr)

    def save_model(self, dec_type):
        name_data = "KVR/" if self.task == '' else "BABI/"
        directory = 'save/mem2seq-' + name_data + str(self.task) + 'HDD' + str(self.hidden_size) + 'BSZ' + str(
            args['batch']) + 'DR' + str(self.dropout) + 'L' + str(self.n_layers) + 'lr' + str(self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    def train_batch(self, input_batches, input_lengths, target_batches,
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio, reset):

        if reset:
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1

        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab, loss_Ptr = 0, 0

        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))

        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(
            max_target_length, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(
            max_target_length, batch_size, input_batches.size(0)))

        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        if use_teacher_forcing:
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(
                    decoder_input, decoder_hidden)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                decoder_input = target_batches[t]  # Chosen word is next input
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()
        else:
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(
                    decoder_input, decoder_hidden)
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vacab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                # get the correspective word in input
                top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(
                    toppi.view(1, -1))).transpose(0, 1)
                next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[
                    i] - 1) else topvi[i].item() for i in range(batch_size)]

                decoder_input = Variable(torch.LongTensor(
                    next_in))  # Chosen word is next input
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()

        # Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(
                0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(
                0, 1).contiguous(),  # -> batch x seq
            target_index.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab + loss_Ptr
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_ptr += loss_Ptr.item()
        self.loss_vac += loss_Vocab.item()

    def evaluate_batch(self, batch_size, input_batches, input_lengths, target_batches, target_lengths, target_index, target_gate, src_plain):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(
            torch.zeros(self.max_r, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(
            self.max_r, batch_size, input_batches.size(0)))
        #all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            #all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()

        p = []
        for elm in src_plain:
            elm_temp = [word_triple[0] for word_triple in elm]
            p.append(elm_temp)

        self.from_whichs = []
        acc_gate, acc_ptr, acc_vac = 0.0, 0.0, 0.0
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(
                decoder_input, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vacab
            topv, topvi = decoder_vacab.data.topk(1)
            all_decoder_outputs_ptr[t] = decoder_ptr
            topp, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(
                toppi.view(1, -1))).transpose(0, 1)
            next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[
                i] - 1) else topvi[i].item() for i in range(batch_size)]

            decoder_input = Variable(torch.LongTensor(
                next_in))  # Chosen word is next input
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            temp = []
            from_which = []
            for i in range(batch_size):
                if(toppi[i].item() < len(p[i]) - 1):
                    temp.append(p[i][toppi[i].item()])
                    from_which.append('p')
                else:
                    ind = topvi[i].item()
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[ind])
                    from_which.append('v')
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
        self.from_whichs = np.array(self.from_whichs)

        # indices = torch.LongTensor(range(target_gate.size(0)))
        # if USE_CUDA: indices = indices.cuda()

        # ## acc pointer
        # y_ptr_hat = all_decoder_outputs_ptr.topk(1)[1].squeeze()
        # y_ptr_hat = torch.index_select(y_ptr_hat, 0, indices)
        # y_ptr = target_index
        # acc_ptr = y_ptr.eq(y_ptr_hat).sum()
        # acc_ptr = acc_ptr.data[0]/(y_ptr_hat.size(0)*y_ptr_hat.size(1))
        # ## acc vocab
        # y_vac_hat = all_decoder_outputs_vocab.topk(1)[1].squeeze()
        # y_vac_hat = torch.index_select(y_vac_hat, 0, indices)
        # y_vac = target_batches
        # acc_vac = y_vac.eq(y_vac_hat).sum()
        # acc_vac = acc_vac.data[0]/(y_vac_hat.size(0)*y_vac_hat.size(1))

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words  # , acc_ptr, acc_vac

    def evaluate(self, dev, avg_best, BLEU=False):
        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        microF1_PRED, microF1_PRED_cal, microF1_PRED_nav, microF1_PRED_wet = 0, 0, 0, 0
        microF1_TRUE, microF1_TRUE_cal, microF1_TRUE_nav, microF1_TRUE_wet = 0, 0, 0, 0
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}

        if args['dataset'] == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_')
                                               for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_')
                                                   for k in item.keys()]
                global_entity_list = list(set(global_entity_list))
        else:
            if int(args["task"]) != 6:
                global_entity_list = entityList(
                    'data/dialog-bAbI-tasks/dialog-babi-kb-all.txt', int(args["task"]))
            else:
                global_entity_list = entityList(
                    'data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt', int(args["task"]))

        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            if args['dataset'] == 'kvr':
                words = self.evaluate_batch(len(data_dev[1]), data_dev[0], data_dev[1],
                                            data_dev[2], data_dev[3], data_dev[4], data_dev[5], data_dev[6])
            else:
                words = self.evaluate_batch(len(data_dev[1]), data_dev[0], data_dev[1],
                                            data_dev[2], data_dev[3], data_dev[4], data_dev[5], data_dev[6])

            acc = 0
            w = 0
            temp_gen = []

            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e == '<EOS>':
                        break
                    else:
                        st += e + ' '
                temp_gen.append(st)
                correct = data_dev[7][i]
                # compute F1 SCORE
                st = st.lstrip().rstrip()
                correct = correct.lstrip().rstrip()
                if args['dataset'] == 'kvr':
                    f1_true, count = self.compute_prf(
                        data_dev[8][i], st.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE += f1_true
                    microF1_PRED += count
                    f1_true, count = self.compute_prf(
                        data_dev[9][i], st.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE_cal += f1_true
                    microF1_PRED_cal += count
                    f1_true, count = self.compute_prf(
                        data_dev[10][i], st.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE_nav += f1_true
                    microF1_PRED_nav += count
                    f1_true, count = self.compute_prf(
                        data_dev[11][i], st.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE_wet += f1_true
                    microF1_PRED_wet += count
                elif args['dataset'] == 'babi' and int(args["task"]) == 6:
                    f1_true, count = self.compute_prf(
                        data_dev[10][i], st.split(), global_entity_list, data_dev[12][i])
                    microF1_TRUE += f1_true
                    microF1_PRED += count

                if args['dataset'] == 'babi':
                    if data_dev[11][i] not in dialog_acc_dict.keys():
                        dialog_acc_dict[data_dev[11][i]] = []
                    if (correct == st):
                        acc += 1
                        dialog_acc_dict[data_dev[11][i]].append(1)
                    else:
                        dialog_acc_dict[data_dev[11][i]].append(0)
                else:
                    if (correct == st):
                        acc += 1
                #    print("Correct:"+str(correct))
                #    print("\tPredict:"+str(st))
                #    print("\tFrom:"+str(self.from_whichs[:,i]))

                w += wer(correct, st)
                ref.append(str(correct))
                hyp.append(str(st))
                ref_s += str(correct) + "\n"
                hyp_s += str(st) + "\n"

            acc_avg += acc / float(len(data_dev[1]))
            wer_avg += w / float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg / float(len(dev)),
                                                            wer_avg / float(len(dev))))

        # dialog accuracy
        if args['dataset'] == 'babi':
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            logging.info("Dialog Accuracy:\t" + str(dia_acc *
                                                    1.0 / len(dialog_acc_dict.keys())))

        if args['dataset'] == 'kvr':
            logging.info("F1 SCORE:\t{}".format(
                microF1_TRUE / float(microF1_PRED)))
            logging.info("\tCAL F1:\t{}".format(
                microF1_TRUE_cal / float(microF1_PRED_cal)))
            logging.info("\tWET F1:\t{}".format(
                microF1_TRUE_wet / float(microF1_PRED_wet)))
            logging.info("\tNAV F1:\t{}".format(
                microF1_TRUE_nav / float(microF1_PRED_nav)))
        elif args['dataset'] == 'babi' and int(args["task"]) == 6:
            logging.info("F1 SCORE:\t{}".format(
                microF1_TRUE / float(microF1_PRED)))

        bleu_score = moses_multi_bleu(
            np.array(hyp), np.array(ref), lowercase=True)
        logging.info("BLEU SCORE:" + str(bleu_score))
        if (BLEU):
            if (bleu_score >= avg_best):
                self.save_model(str(self.name) + str(bleu_score))
                logging.info("MODEL SAVED")
            return bleu_score
        else:
            acc_avg = acc_avg / float(len(dev))
            if (acc_avg >= avg_best):
                self.save_model(str(self.name) + str(acc_avg))
                logging.info("MODEL SAVED")
            return acc_avg

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / \
                float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count
