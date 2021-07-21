FROM python:rc-slim

RUN apt-get update && \
    apt-get install git build-essential gcc-arm-none-eabi libffi-dev libxml2-dev libxslt1-dev python3-pil -y && \
    rm -rf /var/lib/apt/lists/*

ENV BRANCH=master

WORKDIR /opt

#---------------------------------------
# Clone and build sPyNNaker
#---------------------------------------
RUN git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNUtils.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/DataSpecification.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNMachine.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNMan.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/PACMAN.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNFrontEndCommon.git && \
    git clone -b $BRANCH https://github.com/chanokin/sPyNNaker.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNStorageHandlers.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNakerGraphFrontEnd.git
# Clone extra modules
#git clone -b $BRANCH https://github.com/SpiNNakerManchester/sPyNNakerExtraModelsPlugin.git
#git clone -b $BRANCH https://github.com/SpiNNakerManchester/sPyNNakerExternalDevicesPlugin.git
#git clone -b $BRANCH https://github.com/SpiNNakerManchester/PyNNExamples.git

# Install python modules
RUN for mod in `ls /opt`; do \
    cd /opt/$mod/ && python setup.py develop --no-deps; \
    done

# cd ../sPyNNaker8NewModelTemplate
# python setup.py develop --no-deps

#cd ../sPyNNakerExternalDevicesPlugin
#python setup.py develop --no-deps

#cd ../sPyNNakerExtraModelsPlugin
#python setup.py develop --no-deps

# Setup envs
ENV NEURAL_MODELLING_DIRS=/opt/sPyNNaker/neural_modelling
ENV SPINN_DIRS=/opt/spinnaker_tools
ENV SPINN_VERSION=131
ENV PATH=$SPINN_DIRS/tools:$PATH
ENV PERL5LIB=$SPINN_DIRS/tools:$PERL5LIB

#---------------------------------------
# Clone and build low-level tools
#---------------------------------------
RUN git clone -b $BRANCH https://github.com/SpiNNakerManchester/spinnaker_tools.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/spinn_common.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/ybug.git && \
    git clone -b $BRANCH https://github.com/SpiNNakerManchester/spalloc.git
# Source spinnaker tools
RUN cd /opt/spinnaker_tools && \
    make -j 12
# bash ./setup && \

# Make spinn_common
RUN cd /opt/spinn_common && \
    make -j 12 && make install

# RUN cd /opt/SpiNNFrontEndCommon/c_common && \
#     make -j 12

# RUN cd /opt/sPyNNaker/neural_modelling && \
#     make -j 12

# Install dependencies
RUN pip install jsonschema lxml sortedcollections requests \
                requests spalloc appdirs pyyaml csa lazyarray \
                neo pynn quantities numpy
RUN pip install rig rig_c_sa


