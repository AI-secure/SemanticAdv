if [ ! -d pretrain/ ]; then
  mkdir pretrain/
fi

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xJ0yGazHwz24LngOM1FDt1l8r2SMdhb-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xJ0yGazHwz24LngOM1FDt1l8r2SMdhb-" -O ./pretrain/checkpoint_200.pth.tar && rm -rf /tmp/cookies.txt
