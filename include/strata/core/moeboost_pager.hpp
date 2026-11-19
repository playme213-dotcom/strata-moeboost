// include/strata/core/moeboost_pager.hpp - MoEBoost hierarchical expert pager & spillway
// Integrates Gravitize.IGP MoEBoost 3-tier policy directly into the Strata HotRod engine:
// 
// Tier 1: High-Tech Tool Workstation (VRAM GDDR5) - Hot active layer/expert slices
// Tier 2: Hand-Labor Spillway (RAM DDR4) - Dynamic capacity buffer (50% rule)
// Tier 3: Archival Storage (NVMe SSD) - Zero-compute immutable weight storage via pread
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace strata::moeboost {

enum class ExpertLocation {
    VRAM = 1,
    RAM = 2,
    SSD = 3
};

struct ExpertStats {
    std::atomic<uint64_t> invocations{0};
    std::atomic<uint64_t> vram_hits{0};
    std::atomic<uint64_t> ram_hits{0};
    std::atomic<uint64_t> ssd_reads{0};
    std::atomic<uint32_t> current_tier{static_cast<uint32_t>(ExpertLocation::SSD)};
};

class MoEBoostWorkshopManager {
public:
    static MoEBoostWorkshopManager& instance() {
        static MoEBoostWorkshopManager s_mgr;
        return s_mgr;
    }

    void init(int64_t n_layers, int64_t n_experts, uint64_t ram_budget_bytes, uint64_t vram_budget_bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        n_layers_ = n_layers;
        n_experts_ = n_experts;
        ram_budget_ = ram_budget_bytes;
        vram_budget_ = vram_budget_bytes;
        stats_.resize(n_layers * n_experts);
    }

    // Directs routing: returns true if expert can be computed in VRAM, false if diverted to RAM spillway
    bool direct_workstation(int64_t layer, int64_t expert) {
        if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_experts_) return false;
        size_t idx = static_cast<size_t>(layer * n_experts_ + expert);
        stats_[idx].invocations.fetch_add(1, std::memory_order_relaxed);
        
        // Dynamic In-vs-Go: Top hot experts directed to VRAM tool workstation
        uint32_t loc = stats_[idx].current_tier.load(std::memory_order_relaxed);
        if (loc == static_cast<uint32_t>(ExpertLocation::VRAM)) {
            stats_[idx].vram_hits.fetch_add(1, std::memory_order_relaxed);
            return true;
        } else if (loc == static_cast<uint32_t>(ExpertLocation::RAM)) {
            stats_[idx].ram_hits.fetch_add(1, std::memory_order_relaxed);
            return false;
        } else {
            stats_[idx].ssd_reads.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    void set_expert_tier(int64_t layer, int64_t expert, ExpertLocation loc) {
        if (layer >= 0 && expert >= 0 && layer < n_layers_ && expert < n_experts_) {
            size_t idx = static_cast<size_t>(layer * n_experts_ + expert);
            stats_[idx].current_tier.store(static_cast<uint32_t>(loc), std::memory_order_relaxed);
        }
    }

    double global_hit_rate() const {
        uint64_t vram = 0, ram = 0, ssd = 0;
        for (const auto& s : stats_) {
            vram += s.vram_hits.load(std::memory_order_relaxed);
            ram += s.ram_hits.load(std::memory_order_relaxed);
            ssd += s.ssd_reads.load(std::memory_order_relaxed);
        }
        uint64_t total = vram + ram + ssd;
        if (total == 0) return 1.0;
        return static_cast<double>(vram + ram) / static_cast<double>(total);
    }

private:
    MoEBoostWorkshopManager() = default;
    mutable std::mutex mu_;
    int64_t n_layers_ = 0;
    int64_t n_experts_ = 0;
    uint64_t ram_budget_ = 0;
    uint64_t vram_budget_ = 0;
    std::vector<ExpertStats> stats_;
};

} // namespace strata::moeboost
