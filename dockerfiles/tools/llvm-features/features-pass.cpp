#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

static cl::opt<std::string> NewTargetFeatures(
    "new-target-features",
    cl::desc("Specify the new target-features string to apply to all functions"),
    cl::value_desc("feature string"),
    cl::init(""),
    cl::Hidden
);

static cl::opt<std::string> NewTargetCPU(
    "new-target-cpu",
    cl::desc("Specify the new target CPU"),
    cl::value_desc("cpu string"),
    cl::init(""),
    cl::Hidden
);

static cl::opt<std::string> NewTuneCPU(
    "new-tune-cpu",
    cl::desc("Specify the new tune CPU"),
    cl::value_desc("cpu string"),
    cl::init(""),
    cl::Hidden
);

static cl::opt<bool> EraseTuneCPU(
    "erase-tune-cpu-generic",
    cl::desc("Erase tune option for generic CPU"),
    cl::value_desc("tune flag"),
    cl::init(false),
    cl::Hidden
);

static cl::opt<bool> QueryFeatures(
    "query-features",
    cl::desc("Query features"),
    cl::value_desc("query flag"),
    cl::init(false),
    cl::Hidden
);

struct ReplaceTargetFeaturesPass
    : public PassInfoMixin<ReplaceTargetFeaturesPass> {

  std::string TargetFeaturesStr;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {

    bool Changed = false;
    LLVMContext &Ctx = M.getContext();

    std::string old_features;
    std::string old_cpu;
    std::string old_tune;

    Attribute new_features_attr;
    if (!NewTargetFeatures.empty()) {
      new_features_attr = Attribute::get(Ctx, "target-features", NewTargetFeatures);
    }

    Attribute new_features_cpu;
    if (!NewTargetCPU.empty()) {
      new_features_cpu = Attribute::get(Ctx, "target-cpu", NewTargetCPU);
    }

    Attribute new_features_tune;
    if (!NewTuneCPU.empty()) {
      new_features_tune = Attribute::get(Ctx, "tune-cpu", NewTuneCPU);
    }

    for (Function &F : M) {

      if (F.isDeclaration()) {
        continue;
      }

      if (F.hasFnAttribute("target-features")) {

        if (QueryFeatures) {
          old_features = F.getFnAttribute("target-features").getAsString();
        } else {
          F.removeFnAttr("target-features");
          Changed = true;
        }

        if (new_features_attr.isValid()) {
          F.addFnAttr(new_features_attr);
        }
      }

      if (F.hasFnAttribute("target-cpu")) {

        if (QueryFeatures) {
          old_cpu = F.getFnAttribute("target-cpu").getAsString();
        } else {
          F.removeFnAttr("target-cpu");
          Changed = true;
        }

        if (new_features_cpu.isValid()) {
          F.addFnAttr(new_features_cpu);
        }
      }

      if (F.hasFnAttribute("tune-cpu")) {

        if (QueryFeatures) {
          old_tune = F.getFnAttribute("tune-cpu").getAsString();
        } else {

          if (new_features_tune.isValid()) {
            F.removeFnAttr("tune-cpu");
            F.addFnAttr(new_features_tune);
            Changed = true;
          } else if (F.getFnAttribute("tune-cpu").getValueAsString() == "generic" && EraseTuneCPU) {
            F.removeFnAttr("tune-cpu");
            Changed = true;
        }

        }
      }

    }

    if (QueryFeatures) {
      llvm::errs() << old_features << '\n';
      llvm::errs() << old_cpu << '\n';
      llvm::errs() << old_tune << '\n';
    }

    if (Changed) {
      return PreservedAnalyses::none();
    } else {
      return PreservedAnalyses::all();
    }
  }

  static bool isRequired() { return true; }

  static StringRef name() { return "ReplaceTargetFeaturesPass"; }
};

}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION,
    "ReplaceTargetFeatures",
    LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
            ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "replace-target-features") {
            MPM.addPass(ReplaceTargetFeaturesPass());
            return true;
          }
          return false;
        }
      );
    }
  };
}
