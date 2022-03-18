if [ ! -d "/cache" ]; then
  wget https://www.dropbox.com/s/gx9s1vfetkjs5sk/cache.zip?dl=1 -O cache.zip
  unzip cache.zip
fi
if [ ! -f model_intent.ckpt ]; then
  wget https://www.dropbox.com/s/yswtnj0gdtl0sms/model_intent_091022.ckpt?dl=1 -O model_intent.ckpt
fi
if [ ! -f model_slot.ckpt ]; then
  wget https://www.dropbox.com/s/83rka7s6tgwo8j5/model_slot.ckpt?dl=1 -O model_slot.ckpt
fi