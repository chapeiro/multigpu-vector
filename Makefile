# DEBUGFLAGS= -g3 -pg -ggdb 
DEBUGFLAGS= -g -G
# DEBUGFLAGS+= -funsafe-loop-optimizations
# DEBUGFLAGS+= -Wunsafe-loop-optimizations

TFLAGS= -O3 
TFLAGS+= -finline 
TFLAGS+= -funit-at-a-time
TFLAGS+= -march=native
TFLAGS+= -fmerge-all-constants
TFLAGS+= -fmodulo-sched
TFLAGS+= -fmodulo-sched-allow-regmoves
#mstackrealign
TFLAGS+= -funsafe-loop-optimizations
TFLAGS+= -Wunsafe-loop-optimizations
TFLAGS+= -fsched-pressure
TFLAGS+= -fipa-pta
TFLAGS+= -fipa-matrix-reorg
TFLAGS+= -ftree-loop-distribution
TFLAGS+= -ftracer
# TFLAGS+= -funroll-loops
# TFLAGS+= -fwhole-program
# TFLAGS+= -flto

CUDA_INSTALL_PATH?=/usr/local/cuda

# DEBUGFLAGS+=$(TFLAGS)
# OPTFLAGS= $(TFLAGS)
# OPTFLAGS+= -DNDEBUG

CXX=./tools/colornvcc

INCLUDE_PATH=-I. 

# CXXFLAGS= -Wl,--no-as-needed -lpthread -pthread -std=c++11 -Wall -D_GNU_SOURCE 
CXXFLAGS+= -DNQUEUE_WORK
CXXFLAGS+= -O3 -arch=sm_60 -lineinfo --std=c++11 -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED
CXXFLAGS+= -DNTESTMEMCPY -D__STRICT_ANSI__ 
CXXFLAGS+= --maxrregcount=32
CXXFLAGS+= --default-stream per-thread --expt-relaxed-constexpr -rdc=true
CXXFLAGS+= --compiler-options='-fdiagnostics-color=always $(TFLAGS)'

# PROFFLAGS+= -L/usr/local/cuda/lib64 -lnvToolsExt

CXXFLAGS+= $(INCLUDE_PATH)

DBG_DIR=debug
RLS_DIR=release

BIN_ROOT=bin
OBJ_ROOT=obj
SRC_ROOT=src
DEP_ROOT=.depend

BIN_DBG=$(BIN_ROOT)/$(DBG_DIR)/
BIN_RLS=$(BIN_ROOT)/$(RLS_DIR)/

OBJ_DBG=$(OBJ_ROOT)/$(DBG_DIR)/
OBJ_RLS=$(OBJ_ROOT)/$(RLS_DIR)/

DEP_DBG=$(DEP_ROOT)/$(DBG_DIR)/
DEP_RLS=$(DEP_ROOT)/$(RLS_DIR)/

SED_ODD=$(subst /,\/,$(OBJ_DBG))
SED_ORD=$(subst /,\/,$(OBJ_RLS))

SED_DDD=$(subst /,\/,$(DEP_DBG))
SED_DRD=$(subst /,\/,$(DEP_RLS))

EXCLUDE_SOURCES= src/main.cu
EXCLUDE_SOURCES+= src/main2.cu
EXCLUDE_SOURCES+= src/main3.cu
EXCLUDE_SOURCES+= src/main4.cu
EXCLUDE_SOURCES+= src/main5.cu
EXCLUDE_SOURCES+= src/main6.cu
EXCLUDE_SOURCES+= src/output_composer.cu
EXCLUDE_SOURCES+= src/operators/select.cu
EXCLUDE_SOURCES+= src/operators/select2.cu

CXX_SOURCESD= $(shell find $(SRC_ROOT) -name "*.cu")
CXX_SOURCESD:= $(filter-out $(EXCLUDE_SOURCES),$(CXX_SOURCESD))
CXX_SOURCES= $(subst $(SRC_ROOT)/,,$(CXX_SOURCESD))
CXX_OBJECTS= $(CXX_SOURCES:.cu=.o)

OBJ_FILES:=$(addprefix $(OBJ_DBG), $(CXX_OBJECTS)) $(addprefix $(OBJ_RLS), $(CXX_OBJECTS))

# .DEFAULT_GOAL := release
all: debug release

debug:CXXFLAGS+= $(DEBUGFLAGS) $(PROFFLAGS)
release:CXXFLAGS+= $(OPTFLAGS) $(PROFFLAGS)

release:BIN_DIR:= $(BIN_RLS)
release:IMP_DIR:= $(RLS_DIR)
release:OBJ_DIR:= $(OBJ_RLS)
release:CXX_OBJ_D:= $(addprefix $(OBJ_RLS), $(CXX_OBJECTS))

debug:BIN_DIR:= $(BIN_DBG)
debug:IMP_DIR:= $(DBG_DIR)
debug:OBJ_DIR:= $(OBJ_DBG)
debug:CXX_OBJ_D:= $(addprefix $(OBJ_DBG), $(CXX_OBJECTS))

-include $(addprefix $(DEP_DBG), $(CXX_SOURCES:.cu=.d))
-include $(addprefix $(DEP_RLS), $(CXX_SOURCES:.cu=.d))

$(BIN_RLS)engine:$(addprefix $(OBJ_RLS), $(CXX_OBJECTS))
$(BIN_DBG)engine:$(addprefix $(OBJ_DBG), $(CXX_OBJECTS))

release: $(BIN_RLS)engine
debug:   $(BIN_DBG)engine

.PHONY: all debug release 

space= 
#do no remove this lines!!! needed!!!
space+= 

vpath %.o $(subst $(space),:,$(dir $(OBJ_FILES)))
vpath %.cu $(subst $(space),:,$(dir $(CXX_SOURCESD)))


$(sort $(subst //,/,$(dir $(OBJ_FILES)))):
	mkdir -p $@

# %.phpp: %.chpp.l
# 	flex++ --stdout $^ | g++ -x c++ -std=c++11 -o $@ - -lfl -ly

# $(SRC_ROOT)/Board.hpp: $(SRC_ROOT)/Board.chpp $(SRC_ROOT)/lang.phpp
# 	./$(SRC_ROOT)/lang.phpp < $(SRC_ROOT)/Board.chpp > $@

%.o: 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(subst $(OBJ_DIR),$(SRC_ROOT)/,$(@:.o=.cu)) -o $@

%engine:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $^

clean:
	-rm -r $(OBJ_ROOT) $(BIN_ROOT) $(DEP_ROOT)
	mkdir -p $(BIN_DBG) $(BIN_RLS) $(OBJ_DBG) $(OBJ_RLS) $(DEP_DBG) $(DEP_RLS)

$(DEP_DBG)%.d: %.cu Makefile
	@mkdir -p $(@D)
	$(CXX) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MM" $(CPPFLAGS) $(CXXFLAGS) $< | sed -r 's/^(\S+).(\S+):/$(SED_ODD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(<:.cu=.o))) $(SED_DDD)$(subst /,\/,$(<:.cu=.d)): \\\n Makefile \\\n/g' | sed -r 's/(\w)\s+(\w)/\1 \\\n \2/g' | sed '$$s/$$/\\\n | $(SED_ODD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(dir $<)))/g' | sed -r 's/(\w)+\/\.\.\///g' | awk '!x[$$0]++' > $@

$(DEP_RLS)%.d: %.cu Makefile
	@mkdir -p $(@D)
	$(CXX) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MM" $(CPPFLAGS) $(CXXFLAGS) $< | sed -r 's/^(\S+).(\S+):/$(SED_ORD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(<:.cu=.o))) $(SED_DRD)$(subst /,\/,$(<:.cpp=.d)): \\\n Makefile \\\n/g' | sed -r 's/(\w)\s+(\w)/\1 \\\n \2/g' | sed '$$s/$$/\\\n | $(SED_ORD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(dir $<)))/g' | sed -r 's/(\w)+\/\.\.\///g' | awk '!x[$$0]++' > $@
