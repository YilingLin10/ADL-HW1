if [ ! -d "/cache" ]; then
  wget https://www.dropbox.com/s/grv8chkfeynwxme/cache2.zip?dl=1 -O cache.zip
  unzip cache.zip
fi
if [ ! -f model_intent.ckpt ]; then
  wget https://www.dropbox.com/s/2ndvqctnnbyoopj/model_intent_091955.ckpt?dl=1 -O model_intent.ckpt
fi
if [ ! -f model_slot.ckpt ]; then
  wget https://www.dropbox.com/s/o06as0m0ebaco37/model_slot_0.80.ckpt?dl=1 -O model_slot.ckpt
fi