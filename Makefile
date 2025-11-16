# Copyright (C) 2024 Gramine contributors
# SPDX-License-Identifier: BSD-3-Clause

SHELL := /bin/bash

THIS_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
VENV_DIR ?= $(THIS_DIR)/venv_aftune
ENTRYPOINT := $(VENV_DIR)/bin/python3

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

.PHONY: all
all: $(VENV_DIR)/.INSTALLATION_OK aftune.manifest
ifeq ($(SGX),1)
all: aftune.manifest.sgx aftune.sig
endif

.PRECIOUS: $(VENV_DIR)/.INSTALLATION_OK
$(VENV_DIR)/.INSTALLATION_OK:
	python3 -m venv $(VENV_DIR) \
	&& source $(VENV_DIR)/bin/activate \
	&& pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
	&& pip3 install transformers==4.47.1 blake3==1.0.8 tqdm==4.67.1 zstandard==0.22.0 datasets==4.3.0\
	&& pip3 install --no-build-isolation ./aftune_hash/ \
	&& deactivate \
	&& touch $@

aftune.manifest: aftune.manifest.template $(VENV_DIR)/.INSTALLATION_OK
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dentrypoint=$(abspath $(ENTRYPOINT)) \
		-Dvenv_dir=$(abspath $(VENV_DIR)) \
		$< > $@

aftune.manifest.sgx aftune.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: aftune.manifest
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

.PHONY: clean
clean:
	$(RM) *.token *.sig *.manifest.sgx *.manifest

.PHONY: distclean
distclean: clean
	$(RM) -r *.pt result.txt $(VENV_DIR)
