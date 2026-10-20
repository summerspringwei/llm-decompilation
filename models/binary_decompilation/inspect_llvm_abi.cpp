#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static std::string typeName(Type *type) {
  std::string text;
  raw_string_ostream stream(text);
  type->print(stream);
  return stream.str();
}

static json::Array abiAttributes(AttributeSet attributes) {
  json::Array result;
  for (auto kind : {Attribute::SExt, Attribute::ZExt, Attribute::ByVal,
                    Attribute::ByRef, Attribute::StructRet, Attribute::InReg,
                    Attribute::InAlloca, Attribute::Nest, Attribute::SwiftSelf,
                    Attribute::SwiftError}) {
    if (attributes.hasAttribute(kind))
      result.push_back(attributes.getAttribute(kind).getAsString());
  }
  return result;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  LLVMContext context;
  SMDiagnostic error;
  auto module = parseIRFile(argv[1], error, context);
  if (!module) {
    error.print(argv[0], errs());
    return 1;
  }
  json::Object functions;
  for (auto &function : *module) {
    json::Array parameters;
    auto attributes = function.getAttributes();
    for (auto &argument : function.args()) {
      parameters.push_back(json::Object{
          {"type", typeName(argument.getType())},
          {"attributes", abiAttributes(attributes.getParamAttrs(argument.getArgNo()))}});
    }
    json::Array calls;
    for (auto &block : function) {
      for (auto &instruction : block) {
        auto *call = dyn_cast<CallBase>(&instruction);
        if (!call) continue;
        auto *callee = dyn_cast<Function>(call->getCalledOperand()->stripPointerCasts());
        if (!callee) continue;
        auto *type = call->getFunctionType();
        auto callAttributes = call->getAttributes();
        json::Array arguments;
        for (unsigned i = 0; i < type->getNumParams(); ++i)
          arguments.push_back(json::Object{{"type", typeName(type->getParamType(i))},
              {"attributes", abiAttributes(callAttributes.getParamAttrs(i))}});
        calls.push_back(json::Object{{"callee", callee->getName()}, {"abi", json::Object{
            {"return_type", typeName(type->getReturnType())},
            {"return_attributes", abiAttributes(callAttributes.getRetAttrs())},
            {"parameters", std::move(arguments)}, {"variadic", type->isVarArg()},
            {"calling_convention", static_cast<int64_t>(call->getCallingConv())}}}});
      }
    }
    functions[function.getName()] = json::Object{
        {"return_type", typeName(function.getReturnType())},
        {"return_attributes", abiAttributes(attributes.getRetAttrs())},
        {"parameters", std::move(parameters)},
        {"variadic", function.isVarArg()},
        {"calling_convention", static_cast<int64_t>(function.getCallingConv())},
        {"defined", !function.isDeclaration()}, {"calls", std::move(calls)}};
  }
  outs() << formatv("{0:2}\n", json::Value(std::move(functions)));
}
