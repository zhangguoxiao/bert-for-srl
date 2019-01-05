# bert-for-srl

this project is for Semantic role labeling using bert.


part one: Introduction

   For SRL, Since Zhou and Xu (2015), end-to-end system with deep dynamic neural network have been chosen (He et al., 2017; Tan et al., 2017, Peters  et al., 2018).To be specific, Zhou and Xu (2015) introduce two other features (predicate context and region mark) except for the input sequence, while He et al. (2017) and Tan et al.(2017) use a sentence-predicate pair  as the special input. Besides, Tan et al.(2017) choose self-attention as the key component in their architecture instead of LSTMs. Anyway, these end-to-end systems perform better than the traditional models (Pradhan et al., 2013; Täkström et al., 2015).
    Not long ago, the word representation is pre-trained through models including word2vec and glove. With the development of accelerated computing power, more complexed model dealing with complicated contextualized structure has been proposed (elmo,Peters  et al., 2018).  when using ELMo, the f1 score has jumped from 81.4% to 84.6% on the OntoNotes benchmark (Pradhan et al., 2013). 
    Apart from the above feature-based approaches, transfer-learning methods are also popular, which are to pre-train some model architecture on a LM objective before fine-tuning that model for a supervised task.  Using transformer model, Devlin et al. (2018) propose a new language representation mode : bert. the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-theart models for a wide range of task.The object of this project is to continue the original work, and use the pre-trained BERT for SRL. 
    The relative positional information for each word can be learned automatically with transformer model. Thus, it is sufficient to annotate the target in the word sequence. Here, in this study, we choose two position indicators to annotate the target predicate. A sequence with n predicates is processed n times. Each time, the target predicate is annotated with two position indicators. 
 

part two: experiments

  All the following experiments are based on the English OntoNotes dataset (Pradhan et al., 2013). 
  Tensorflow 1.12 and cuda 9.0 are used on GTX 1080 Ti. The pretrained model of our experiments are bert-based model "cased_L-12_H-768_A-12" with 12-layer, 768-hidden, 12-heads , 110M parameters. The large model doesn't work on  GTX 1080 Ti.
  To run the code, the train/dev/test dataset need to be processed as the following format: each line with two parts, one is BIO tags, one is the raw sentence with an annotated predicate, the two parts are splitted by "\t". For example:
  
    "O O B_ARG1 I_ARG1 E_ARG1 O V O B_ARGM-TMP I_ARGM-TMP I_ARGM-TMP I_ARGM-TMP I_ARGM-TMP I_ARGM-TMP I_ARGM-TMP I_ARGM-TMP I_ARGM-TMP E_ARGM-TMP O	There 's only one month </s left s/> before the opening of Hong Kong Disneyland on September 12 ."
    
  The default tagging is BIO, you can also use BIESO tagging strategy, if so, you need to change the method get_labels() of SrlProcessor in bert_lstm_crf_srl.py.
  Using the default setting : bert + crf.
  
      export TRAINED_CLASSIFIER=/your/model/dir
      export BERT_MODEL_DIR=/bert/pretrained/model/dir
      export DATA_DIR=/data/dir/
      python bert_lstm_crf_srl.py  --task_name="srl" \
             --do_train=True  \
              --do_eval=True  \
              --do_predict=True  \
             --data_dir=$DATA_DIR  \
             --vocab_file=$BERT_MODEL_DIR/vocab.txt \
             --bert_config_file==$BERT_MODEL_DIR/bert_config.json \
             --init_checkpoint==$BERT_MODEL_DIR/bert_model.ckpt \
             --max_seq_length=128 \
             --train_batch_size=32   \
             --learning_rate=3e-5   \
             --num_train_epochs=3.0  \
             --output_dir=$TRAINED_CLASSIFIER/ 
             
  If you want to add lstm layer:
  
         export TRAINED_CLASSIFIER=/your/model/dir
         export BERT_MODEL_DIR=/bert/pretrained/model/dir
         export DATA_DIR=/data/dir/
         python bert_lstm_crf_srl.py  --task_name="srl" \
                --do_train=True  \
                 --do_eval=True  \
                 --do_predict=True  \
                --data_dir=$DATA_DIR  \
                --vocab_file=$BERT_MODEL_DIR/vocab.txt \
                --bert_config_file==$BERT_MODEL_DIR/bert_config.json \
                --init_checkpoint==$BERT_MODEL_DIR/bert_model.ckpt \
                --max_seq_length=128 \
                --train_batch_size=32   \
                --learning_rate=3e-5   \
                --num_train_epochs=3.0  \
                 --add_lstm=True  \
                  --output_dir=$TRAINED_CLASSIFIER/ 
   To get the right f1 score, you need to run another file:
   
           python evaluate_unit.py --output_dir /the/predicted/dir --data_dir /the/test/file/dir --vocab_file /vocab/dir
   
   The full results are as follows, you can find the special name "all", "all  presition: 0.84863  recall: 0.85397   fvalue: 0.85129"
   
             ******************************************
          C-ARG2  presition: 0.25000  recall: 0.10526   fvalue: 0.14815
          C-ARGM-COM  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARG1  presition: 0.86449  recall: 0.89204   fvalue: 0.87805
          C-ARGM-TMP  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARG3  presition: 0.67978  recall: 0.58173   fvalue: 0.62694
          ARGM-PRX  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARGM-MNR  presition: 0.62500  recall: 0.45455   fvalue: 0.52632
          C-ARGM-ADJ  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARGM-ADV  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARGM-COM  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARGM-MNR  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARG0  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-TMP  presition: 0.83824  recall: 0.88497   fvalue: 0.86097
          R-ARGM-GOL  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARG5  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARGM-CAU  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-DSP  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARG3  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARGM-LOC  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARGM-MOD  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-MNR  presition: 0.69346  recall: 0.71853   fvalue: 0.70577
          ARGM-ADJ  presition: 0.55128  recall: 0.50391   fvalue: 0.52653
          ARG4  presition: 0.77228  recall: 0.78788   fvalue: 0.78000
          C-ARGM-DSP  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          all  presition: 0.84863  recall: 0.85397   fvalue: 0.85129
          ARGM-REC  presition: 1.00000  recall: 0.25714   fvalue: 0.40909
          ARGM-PRR  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARGM-DIS  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARG4  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-NEG  presition: 0.90909  recall: 0.96642   fvalue: 0.93688
          R-ARG5  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-GOL  presition: 0.25000  recall: 0.41250   fvalue: 0.31132
          ARGM-PNC  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARG1  presition: 0.88026  recall: 0.89463   fvalue: 0.88738
          R-ARG0  presition: 0.93312  recall: 0.91706   fvalue: 0.92502
          C-ARGM-PRP  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARGM-EXT  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARGM-ADV  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARGM-EXT  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-PRP  presition: 0.60036  recall: 0.71552   fvalue: 0.65290
          R-ARGM-PRP  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGA  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARGM-PNC  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARGM-CAU  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-ADV  presition: 0.67350  recall: 0.63141   fvalue: 0.65178
          ARG0  presition: 0.91667  recall: 0.92111   fvalue: 0.91888
          ARGM-COM  presition: 0.32353  recall: 0.73333   fvalue: 0.44898
          ARGM-LVB  presition: 0.82278  recall: 0.90278   fvalue: 0.86093
          ARGM-CAU  presition: 0.72143  recall: 0.77892   fvalue: 0.74907
          R-ARGM-DIR  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARG2  presition: 0.52500  recall: 0.44681   fvalue: 0.48276
          R-ARGM-LOC  presition: 0.56075  recall: 0.86957   fvalue: 0.68182
          C-ARGM-MOD  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-DIS  presition: 0.83903  recall: 0.79242   fvalue: 0.81506
          R-ARG4  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-LOC  presition: 0.72954  recall: 0.71825   fvalue: 0.72385
          ARGM-MOD  presition: 0.97805  recall: 0.97542   fvalue: 0.97673
          C-ARGM-NEG  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-DIR  presition: 0.59913  recall: 0.62358   fvalue: 0.61111
          C-ARGM-DIR  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          ARGM-EXT  presition: 0.43446  recall: 0.65169   fvalue: 0.52135
          ARG2  presition: 0.83115  recall: 0.82796   fvalue: 0.82955
          ARGM-PRD  presition: 0.30631  recall: 0.23611   fvalue: 0.26667
          R-ARGM-PRD  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          R-ARG3  presition: 0.00000  recall: 0.00000   fvalue: 0.00000
          C-ARG1  presition: 0.43151  recall: 0.27632   fvalue: 0.33690
          R-ARGM-TMP  presition: 0.61905  recall: 0.81250   fvalue: 0.70270
           
   Using the default setting,  The init learning rates are different for parameters with namescope "bert" and parameters with namescope "lstm-crf". You can change it through setting lr_2 = lr_gen(0.001) in line 73 of optimization.py.
   
   
part three: results
    
   For the experiments, when adding lstm , no better results has come out. For the different tagging strategy, no significant difference has been observed. The split learning strategy is useful.

   For BIO + 3epoch  + crf with no split learning strategy:
    
             all  presition: 0.84863  recall: 0.85397   fvalue: 0.85129
     
   For BIO + 3epoch  + crf with  split learning strategy:
   
             all  presition: 0.85558  recall: 0.85692   fvalue: 0.85625
             
   For BIOES + 3epoch  + crf with split learning strategy:
   
             all  presition: 0.85523  recall: 0.85906   fvalue: 0.85714
             
   For BIOES + 5epoch  + crf with split learning strategy:
   
             all  presition: 0.85895  recall: 0.86071   fvalue: 0.85983
  
  
 references:
 
    https://github.com/google-research/bert
    https://github.com/macanv/BERT-BiLSTM-CRF-NER
    Jie Zhou and Wei Xu. 2015. End-to-end learning of semantic role labeling using recurrent neural networks. In Proc. of ACL.
    Luheng He, Kenton Lee, Mike Lewis and Luke Zettlemoyer. 2017. Deep semantic role labeling: What works and whats next. In Pro. of ACL.
    Zhixing Tan, Mingxuan Wang, Jun Xie, Yidong Chen, and Xiaodong Shi. 2017. Deep Semantic Role Labeling with Self-Attention. arXiv:1712.01586v1.
    Oscar Täkström, Kuzman Ganchev, and Dipanjan Das. 2015. Efficient inference and structured learning for semantic role labeling. Transactions of the Association for Computational Linguistics,  3:29–41.
    Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Hwee Tou Ng, Anders Björkelund, Olga Uryupina, Yuchen Zhang, and Zhi Zhong. 2013. Towards robust linguistic analysis using ontonotes. In Proc. of COLING.
    Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In NAACL.
    Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.
    
  

    
    


