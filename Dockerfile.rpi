FROM resin/rpi-raspbian:stretch
MAINTAINER Vallard

RUN apt-get update && \
        apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libavformat-dev \
        libpq-dev \
        libgtk2.0-dev \
        pkg-config \
	      python3 \
	      python3-pip \
      	python3-setuptools \
	      python3-dev \
	      libblas-dev \
	      liblapack-dev \
	      libhdf5-dev \
	      python3-h5py \
	      python3-scipy  \
	      python3-pil

RUN pip3 install numpy \
        keras==2.1.2  \
	paho-mqtt \
        h5py==2.7.1

# install Tensorflow
ADD tensorflow-1.1.0-cp35-cp35m-linux_armv7l.whl . 
RUN pip3 install ./tensorflow-1.1.0-cp35-cp35m-linux_armv7l.whl

# install opencv
WORKDIR /
RUN wget https://github.com/opencv/opencv/archive/3.3.0.zip \
&& unzip 3.3.0.zip \
&& mkdir /opencv-3.3.0/cmake_binary \
&& cd /opencv-3.3.0/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make install \
&& rm /3.3.0.zip \
&& rm -r /opencv-3.3.0


ADD src/ /app
WORKDIR /app
CMD ["python3", "yolo-pi.py"]
