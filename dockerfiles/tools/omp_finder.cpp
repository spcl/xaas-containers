#include "clang/AST/AST.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

class OMPDetector : public MatchFinder::MatchCallback {
public:
  void run(const MatchFinder::MatchResult &Result) override { Found = true; }

  bool isFound() const { return Found; }

private:
  bool Found = false;
};

int main(int argc, const char **argv) {

  llvm::cl::OptionCategory Category("omp-finder");
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, Category);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &OptionsParser = ExpectedParser.get();

  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  OMPDetector Detector;
  MatchFinder Finder;

  // Executable Directive should be a parent of all directives
  // https://clang.llvm.org/doxygen/classclang_1_1OMPExecutableDirective.html
  StatementMatcher OMPMatcher = ompExecutableDirective().bind("omp");

  Finder.addMatcher(OMPMatcher, &Detector);

  Tool.run(newFrontendActionFactory(&Finder).get());

  llvm::outs() << (Detector.isFound() ? "XAAS_OMP_FOUND" : "XAAS_OMP_NOTFOUND");

  return 0;
}
