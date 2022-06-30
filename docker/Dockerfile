FROM tensorflow/tensorflow:2.2.0
RUN apt update && apt upgrade -y && DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt install -y --no-install-recommends git curl zsh vim less python3-tk feh
RUN pip install matplotlib python-box tqdm scipy tensorflow_probability==0.10.1
ADD docker/tf_prob_patch_0.10.1 /src/tf_prob_patch_0.10.1
RUN patch -p2 -d /usr/local/lib/python3.6/dist-packages/tensorflow_probability < /src/tf_prob_patch_0.10.1
RUN mkdir -p /root/.config/matplotlib && echo "backend: TKAgg" > /root/.config/matplotlib/matplotlibrc
RUN mkdir /src/cpmp
ADD setup.py /src/cpmp
ADD requirements.txt /src/cpmp
ADD opt_pmp_utils /src/cpmp/opt_pmp_utils
RUN pip install -e /src/cpmp
ADD quant_experiment /src/cpmp/quant_experiment