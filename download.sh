if [ ! -f model_intent.ckpt ]; then
  wget https://www.dropbox.com/s/6s2pajajcmr9edv/model_intent.ckpt?dl=1 -O model_intent.ckpt
fi
if [ ! -f model_slot.ckpt ]; then
  wget https://www.dropbox.com/s/bvscyj7jwqyyhka/model_slot.ckpt?dl=1 -O model_slot.ckpt
fi
