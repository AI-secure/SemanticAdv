if [ ! -d pretrain_models/ ]; then
  mkdir pretrain_models/
fi

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JuEaOcAZRl440jKglowewYo6J6o4V2Cc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JuEaOcAZRl440jKglowewYo6J6o4V2Cc" -O ./pretrain_models/res101_softmax.pth.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yceVJ-yqMQnnP8Q2EIKjPGXiftbRUUL-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yceVJ-yqMQnnP8Q2EIKjPGXiftbRUUL-" -O ./pretrain_models/200000-G.ckpt && rm -rf /tmp/cookies.txt
