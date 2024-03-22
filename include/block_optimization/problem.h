#pragma once

#include "block.h"
#include "processing_chain.h"

#include <map>


namespace block_optimization {

template <typename T>
class Problem {
public:
    void addBlock(std::string name, BlockPtr<T> block) {
        if (blocks_.find(name) != blocks_.end()) {
            throw std::runtime_error("Block " + name + " already exists");
        }
        blocks_[name] = block;
    }

    ProcessingChainPtr<T> makeChain(const std::vector<std::string> chain) {
        ProcessingChainPtr<T> chainPtr = std::make_shared<ProcessingChain<T>>();
        for (const std::string& blockName : chain) {
            chainPtr->appendBlock(blocks_.at(blockName));
        }
    }

    BlockPtr<T>& block(std::string name) {
        return blocks_.at(name);
    }

protected:
    std::map<std::string, BlockPtr<T>> blocks_;

private:
};

} // namespace block_optimization
