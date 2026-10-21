#include "serve/model_registry.h"

#include "artifact/reader.h"
#include "runtime/engine/host_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

using Clock = std::chrono::steady_clock;

// Events kept for GET /models/sse subscribers that fall behind.
constexpr std::size_t kKeptEvents = 256;

} // namespace

const char* model_state_name(ModelState state) noexcept {
    switch (state) {
    case ModelState::Unloaded:
        return "unloaded";
    case ModelState::Loading:
        return "loading";
    case ModelState::Loaded:
        return "loaded";
    case ModelState::Sleeping:
        return "sleeping";
    case ModelState::Unloading:
        return "unloading";
    }
    return "unloaded";
}

ModelRegistry::ModelRegistry(std::vector<ModelDefinition> models, RegistryPolicy policy,
                             ModelServiceFactory factory, bool router)
    : policy_(policy), factory_(std::move(factory)), router_(router) {
    if (models.empty()) { throw std::invalid_argument("a server needs at least one model"); }
    if (!router && models.size() != 1) {
        throw std::invalid_argument("single-model mode serves exactly one model");
    }
    for (ModelDefinition& definition : models) {
        models_.push_back(Slot{.definition = std::move(definition)});
    }
    // A router also watches for models whose Engine failed, which it unloads.
    if (router_ || policy_.sleep_idle.count() > 0 || policy_.unload_idle.count() > 0) {
        idle_thread_ = std::thread([this] { idle_loop(); });
    }
}

ModelRegistry::~ModelRegistry() {
    stop();
    if (idle_thread_.joinable()) { idle_thread_.join(); }
    std::unique_lock lock(mutex_);
    for (Slot& slot : models_) {
        std::unique_ptr<ModelService> service = std::move(slot.service);
        lock.unlock();
        service.reset();
        lock.lock();
    }
}

void ModelRegistry::stop() noexcept {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    changed_.notify_all();
}

std::optional<std::size_t> ModelRegistry::find_locked(std::string_view model) const {
    if (model.empty()) {
        if (models_.size() == 1) { return 0; }
        // The most recently used loaded model serves a request that names none.
        std::optional<std::size_t> best;
        for (std::size_t index = 0; index < models_.size(); ++index) {
            const Slot& slot = models_[index];
            if (slot.state != ModelState::Loaded && slot.state != ModelState::Sleeping) {
                continue;
            }
            if (!best || slot.last_used > models_[*best].last_used) { best = index; }
        }
        return best;
    }
    for (std::size_t index = 0; index < models_.size(); ++index) {
        const ModelDefinition& definition = models_[index].definition;
        if (definition.id == model ||
            std::find(definition.aliases.begin(), definition.aliases.end(), model) !=
                definition.aliases.end()) {
            return index;
        }
    }
    return std::nullopt;
}

std::size_t ModelRegistry::require_locked(std::string_view model) const {
    const auto index = find_locked(model);
    if (!index) {
        if (model.empty()) {
            throw ModelUnknown("the request names no model and none is loaded; set \"model\"");
        }
        throw ModelUnknown("model '" + std::string(model) + "' not found");
    }
    return *index;
}

std::optional<std::string> ModelRegistry::resolve(std::string_view model) const {
    std::lock_guard lock(mutex_);
    const auto index = find_locked(model);
    if (!index) { return std::nullopt; }
    return models_[*index].definition.id;
}

std::size_t ModelRegistry::loaded_count_locked() const {
    return static_cast<std::size_t>(
        std::count_if(models_.begin(), models_.end(), [](const Slot& s) {
            return s.state == ModelState::Loaded || s.state == ModelState::Loading;
        }));
}

void ModelRegistry::publish_locked(std::size_t index) {
    const Slot& slot = models_[index];
    events_.push_back(ModelEvent{.model  = slot.definition.id,
                                 .state  = slot.state,
                                 .failed = slot.failed,
                                 .error  = slot.last_error});
    while (events_.size() > kKeptEvents) {
        events_.pop_front();
        ++first_event_;
    }
    changed_.notify_all();
}

void ModelRegistry::put_to_sleep_locked(std::unique_lock<std::mutex>& lock, std::size_t index) {
    Slot& slot                  = models_[index];
    ModelService* const service = slot.service.get();
    slot.state                  = ModelState::Unloading;
    publish_locked(index);
    lock.unlock();
    std::exception_ptr failure;
    try {
        service->suspend();
    } catch (...) { failure = std::current_exception(); }
    lock.lock();
    if (failure) {
        // Still resident: a request may have arrived in between. It stays loaded.
        slot.state = ModelState::Loaded;
        publish_locked(index);
        std::rethrow_exception(failure);
    }
    slot.state = ModelState::Sleeping;
    publish_locked(index);
}

void ModelRegistry::unload_locked(std::unique_lock<std::mutex>& lock, std::size_t index) {
    Slot& slot = models_[index];
    slot.state = ModelState::Unloading;
    publish_locked(index);
    std::unique_ptr<ModelService> service = std::move(slot.service);
    lock.unlock();
    // Destroying the service stops its Engine, which saves what it persists, and frees the model's
    // device and host memory.
    service.reset();
    lock.lock();
    slot.state = ModelState::Unloaded;
    publish_locked(index);
}

bool ModelRegistry::make_room_locked(std::unique_lock<std::mutex>& lock, std::size_t keep,
                                     Clock::time_point deadline) {
    for (;;) {
        if (stopping_) { return false; }
        if (policy_.models_max == 0 || loaded_count_locked() < policy_.models_max) { return true; }
        std::optional<std::size_t> victim;
        for (std::size_t index = 0; index < models_.size(); ++index) {
            const Slot& slot = models_[index];
            if (index == keep || slot.state != ModelState::Loaded || slot.leases != 0) { continue; }
            if (!victim || slot.last_used < models_[*victim].last_used) { victim = index; }
        }
        if (victim) {
            if (models_[*victim].service->suspendable()) {
                try {
                    put_to_sleep_locked(lock, *victim);
                } catch (...) {
                    // Its Engine is still settling the request that just let go of it; look again
                    // shortly rather than spin.
                    const auto retry =
                        std::min(deadline, Clock::now() + std::chrono::milliseconds(50));
                    if (changed_.wait_until(lock, retry) == std::cv_status::timeout &&
                        Clock::now() >= deadline) {
                        return false;
                    }
                }
                continue;
            }
            unload_locked(lock, *victim);
            continue;
        }
        if (changed_.wait_until(lock, deadline) == std::cv_status::timeout) { return false; }
    }
}

std::optional<std::uint64_t> ModelRegistry::device_free_bytes() {
    std::size_t free_bytes = 0;
    std::size_t total      = 0;
    if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess) {
        (void)cudaGetLastError();
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(free_bytes);
}

std::optional<std::uint64_t> ModelRegistry::model_device_bytes(const ModelDefinition& model) {
    // The artifact's stored payload plus the per-stage and head budgets the load planner reserves
    // on top of it (models/qwen3_5/load.cpp:141-143): the same figures that decide whether a load
    // fits there, so a requirement that fits in the planner fits here.
    constexpr std::uint64_t kStageFixedBytes     = 1536ULL << 20;
    constexpr std::uint64_t kHeadStageFixedBytes = 1024ULL << 20;
    try {
        const artifact::Reader reader(model.options.artifact_path);
        return reader.directory().payload_bytes + kStageFixedBytes + kHeadStageFixedBytes;
    } catch (const std::exception&) { return std::nullopt; }
}

std::uint64_t ModelRegistry::model_host_bytes(const ModelDefinition& model) {
    // KVMem pins a host payload budget per Engine and this is the one reservation a router can
    // double without any device allocation getting smaller: two KVMem models at --kvmem-host-mib
    // 5120 hold ten gibibytes of host between them, whatever the card is doing. Its disk tier is
    // a quota rather than a pin, so it is not counted.
    return model.options.kvmem.host_bytes;
}

bool ModelRegistry::device_has_room_locked(std::size_t index, std::string* why) const {
    if (!policy_.enforce_device_room) { return true; }
    const ModelDefinition& definition = models_[index].definition;
    if (const std::optional<std::uint64_t> free_bytes = device_free_bytes()) {
        if (const std::optional<std::uint64_t> needed = model_device_bytes(definition)) {
            if (*needed > *free_bytes) {
                if (why != nullptr) {
                    char line[256];
                    std::snprintf(line, sizeof(line),
                                  "'%s' needs %llu MiB on the device and %llu MiB is free",
                                  definition.id.c_str(),
                                  static_cast<unsigned long long>((*needed + (1ULL << 20) - 1) >> 20),
                                  static_cast<unsigned long long>(*free_bytes >> 20));
                    *why = line;
                }
                return false;
            }
        }
    }
    // The host budget is checked the same way and for the same reason: releasing an idle model
    // returns both, so `evict_idle_locked` is the right response to either shortfall.
    const std::uint64_t host_needed = model_host_bytes(definition);
    if (host_needed != 0) {
        if (const std::optional<std::uint64_t> host_free = runtime::available_host_memory_bytes()) {
            if (host_needed > *host_free) {
                if (why != nullptr) {
                    char line[256];
                    std::snprintf(line, sizeof(line),
                                  "'%s' pins %llu MiB of host KVMem and %llu MiB is available",
                                  definition.id.c_str(),
                                  static_cast<unsigned long long>((host_needed + (1ULL << 20) - 1) >> 20),
                                  static_cast<unsigned long long>(*host_free >> 20));
                    *why = line;
                }
                return false;
            }
        }
    }
    return true;
}

bool ModelRegistry::evict_idle_locked(std::unique_lock<std::mutex>& lock, std::size_t keep) {
    std::optional<std::size_t> victim;
    for (std::size_t index = 0; index < models_.size(); ++index) {
        const Slot& slot = models_[index];
        if (index == keep || slot.state != ModelState::Loaded || slot.leases != 0) { continue; }
        if (!victim || slot.last_used < models_[*victim].last_used) { victim = index; }
    }
    if (!victim) { return false; }
    // A suspended model keeps its host caches and gives the device memory back, so it is the
    // cheaper eviction where it is available; otherwise the Engine goes.
    if (models_[*victim].service->suspendable()) {
        put_to_sleep_locked(lock, *victim);
    } else {
        unload_locked(lock, *victim);
    }
    return true;
}

void ModelRegistry::make_loaded_locked(std::unique_lock<std::mutex>& lock, std::size_t index) {
    const Clock::time_point deadline = Clock::now() + policy_.room_timeout;
    for (;;) {
        Slot& slot = models_[index];
        if (stopping_) { throw ModelLoadFailed("the server is stopping"); }
        if (slot.state == ModelState::Loaded) { return; }
        if (slot.state == ModelState::Loading || slot.state == ModelState::Unloading) {
            if (changed_.wait_until(lock, deadline) == std::cv_status::timeout) {
                throw ModelLoadFailed("model '" + slot.definition.id + "' did not become ready");
            }
            continue;
        }
        const bool waking = slot.state == ModelState::Sleeping;
        if (!make_room_locked(lock, index, deadline)) {
            throw ModelLoadFailed("no room to load model '" + slot.definition.id +
                                  "': every loaded model is busy");
        }
        // `models_max` says how many models may be loaded; it does not say they fit together. Ask
        // the device before building the Engine: an allocation failure inside it arrives as a bare
        // cudaErrorMemoryAllocation from whichever buffer happened to be first, and the request that
        // asked for the model cannot tell a full card from a broken artifact. Keep giving back
        // idle models while any are left -- `models_max` is not the binding constraint here, memory
        // is -- and report the shortfall only when nothing else can be released.
        {
            std::string shortfall;
            while (!device_has_room_locked(index, &shortfall)) {
                if (!evict_idle_locked(lock, index)) {
                    throw ModelLoadFailed("no room for model " + shortfall +
                                          ", and every other loaded model is busy so nothing "
                                          "more can be released");
                }
            }
        }
        if (slot.state != ModelState::Unloaded && slot.state != ModelState::Sleeping) { continue; }
        slot.state = ModelState::Loading;
        publish_locked(index);
        ModelService* const sleeping     = waking ? slot.service.get() : nullptr;
        const ModelDefinition definition = slot.definition;
        lock.unlock();
        std::unique_ptr<ModelService> created;
        std::exception_ptr failure;
        try {
            if (sleeping != nullptr) {
                sleeping->resume();
            } else {
                created = factory_(definition);
            }
        } catch (...) { failure = std::current_exception(); }
        lock.lock();
        Slot& done = models_[index];
        if (failure) {
            done.state  = waking ? ModelState::Sleeping : ModelState::Unloaded;
            done.failed = true;
            try {
                std::rethrow_exception(failure);
            } catch (const std::exception& error) { done.last_error = error.what(); } catch (...) {
                done.last_error = "unknown error";
            }
            publish_locked(index);
            throw ModelLoadFailed("model '" + definition.id +
                                  "' failed to load: " + done.last_error);
        }
        if (created) { done.service = std::move(created); }
        done.state  = ModelState::Loaded;
        done.failed = false;
        done.last_error.clear();
        done.last_used = Clock::now();
        publish_locked(index);
        return;
    }
}

ModelRegistry::Lease ModelRegistry::acquire(std::string_view model, std::optional<bool> autoload) {
    std::unique_lock lock(mutex_);
    const std::size_t index = require_locked(model);
    Slot& slot              = models_[index];
    if (slot.state == ModelState::Sleeping && slot.held) {
        throw ModelNotLoaded("model '" + slot.definition.id + "' is suspended");
    }
    if (slot.state != ModelState::Loaded) {
        if (!autoload.value_or(policy_.autoload)) {
            throw ModelNotLoaded("model '" + slot.definition.id + "' is " +
                                 model_state_name(slot.state) + " and autoload is off");
        }
        make_loaded_locked(lock, index);
    }
    Slot& ready = models_[index];
    ++ready.leases;
    ready.last_used = Clock::now();
    return Lease(*this, index, true);
}

void ModelRegistry::release(std::size_t index, bool use) noexcept {
    {
        std::lock_guard lock(mutex_);
        Slot& slot = models_[index];
        if (slot.leases != 0) { --slot.leases; }
        if (use) { slot.last_used = Clock::now(); }
    }
    changed_.notify_all();
}

std::optional<ModelRegistry::Lease> ModelRegistry::observe(std::string_view model) {
    std::lock_guard lock(mutex_);
    const std::size_t index = require_locked(model);
    Slot& slot              = models_[index];
    if (slot.state != ModelState::Loaded && slot.state != ModelState::Sleeping) {
        return std::nullopt;
    }
    ++slot.leases;
    return Lease(*this, index, false);
}

void ModelRegistry::load(std::string_view model) {
    std::unique_lock lock(mutex_);
    const std::size_t index = require_locked(model);
    models_[index].held     = false;
    make_loaded_locked(lock, index);
}

void ModelRegistry::unload(std::string_view model) {
    std::unique_lock lock(mutex_);
    const std::size_t index = require_locked(model);
    for (;;) {
        Slot& slot = models_[index];
        if (slot.state == ModelState::Unloaded) { return; }
        if (slot.state == ModelState::Loading || slot.state == ModelState::Unloading ||
            slot.leases != 0) {
            changed_.wait(lock);
            continue;
        }
        unload_locked(lock, index);
        return;
    }
}

void ModelRegistry::sleep(std::string_view model, bool hold, bool wait) {
    std::unique_lock lock(mutex_);
    const std::size_t index = require_locked(model);
    for (;;) {
        Slot& slot = models_[index];
        if (slot.state == ModelState::Sleeping || slot.state == ModelState::Unloaded) {
            slot.held = hold;
            publish_locked(index);
            return;
        }
        if (slot.state != ModelState::Loaded || slot.leases != 0) {
            if (!wait) { throw ModelBusy("model '" + slot.definition.id + "' is in use"); }
            changed_.wait(lock);
            continue;
        }
        if (!slot.service->suspendable()) {
            throw ModelNotLoaded("model '" + slot.definition.id +
                                 "' was not started with --model-suspend");
        }
        put_to_sleep_locked(lock, index);
        models_[index].held = hold;
        return;
    }
}

ModelStatusSnapshot ModelRegistry::snapshot_locked(const Slot& slot) const {
    return ModelStatusSnapshot{.id         = slot.definition.id,
                               .aliases    = slot.definition.aliases,
                               .state      = slot.state,
                               .arguments  = slot.definition.arguments,
                               .failed     = slot.failed,
                               .last_error = slot.last_error,
                               .leases     = slot.leases,
                               .last_used  = slot.last_used,
                               .held       = slot.held};
}

std::vector<ModelStatusSnapshot> ModelRegistry::status() const {
    std::lock_guard lock(mutex_);
    std::vector<ModelStatusSnapshot> out;
    out.reserve(models_.size());
    for (const Slot& slot : models_) { out.push_back(snapshot_locked(slot)); }
    return out;
}

std::optional<ModelStatusSnapshot> ModelRegistry::status(std::string_view model) const {
    std::lock_guard lock(mutex_);
    const auto index = find_locked(model);
    if (!index) { return std::nullopt; }
    return snapshot_locked(models_[*index]);
}

std::uint64_t ModelRegistry::events(std::uint64_t cursor, std::chrono::milliseconds wait,
                                    std::vector<ModelEvent>& out) {
    std::unique_lock lock(mutex_);
    const auto next = [&] { return first_event_ + events_.size(); };
    if (cursor == 0) { cursor = next(); }
    if (cursor >= next() && !stopping_) {
        changed_.wait_for(lock, wait, [&] { return stopping_ || cursor < next(); });
    }
    if (cursor < first_event_) { cursor = first_event_; }
    for (std::uint64_t position = cursor; position < next(); ++position) {
        out.push_back(events_[static_cast<std::size_t>(position - first_event_)]);
    }
    return next();
}

void ModelRegistry::for_each_service(
    const std::function<void(const std::string&, GenerationService&)>& visit) {
    std::lock_guard lock(mutex_);
    for (Slot& slot : models_) {
        if (slot.service) { visit(slot.definition.id, slot.service->service()); }
    }
}

void ModelRegistry::load_startup_models() {
    std::vector<std::string> startup;
    {
        std::lock_guard lock(mutex_);
        for (const Slot& slot : models_) {
            if (!router_ || slot.definition.load_on_startup) {
                startup.push_back(slot.definition.id);
            }
        }
    }
    for (const std::string& id : startup) { load(id); }
}

void ModelRegistry::idle_loop() {
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        changed_.wait_for(lock, std::chrono::seconds(1));
        if (stopping_) { break; }
        const Clock::time_point now = Clock::now();
        for (std::size_t index = 0; index < models_.size(); ++index) {
            Slot& slot = models_[index];
            if (slot.leases != 0) { continue; }
            const auto idle = now - slot.last_used;
            try {
                if (router_ && slot.state == ModelState::Loaded && slot.service->failed()) {
                    slot.failed = true;
                    slot.last_error =
                        "the model's Engine failed; it loads again on the next request";
                    unload_locked(lock, index);
                } else if (slot.state == ModelState::Loaded && policy_.sleep_idle.count() > 0 &&
                           idle >= policy_.sleep_idle && slot.service->suspendable()) {
                    put_to_sleep_locked(lock, index);
                } else if ((slot.state == ModelState::Loaded ||
                            slot.state == ModelState::Sleeping) &&
                           policy_.unload_idle.count() > 0 && idle >= policy_.unload_idle) {
                    unload_locked(lock, index);
                }
            } catch (...) {
                // A busy model stays as it is; the next pass looks again.
            }
        }
    }
}

ModelRegistry::Lease::Lease(ModelRegistry& owner, std::size_t index, bool use) noexcept
    : owner_(&owner), index_(index), use_(use) {}

ModelRegistry::Lease::~Lease() {
    if (owner_ != nullptr) { owner_->release(index_, use_); }
}

ModelRegistry::Lease::Lease(Lease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_), use_(other.use_) {}

ModelRegistry::Lease& ModelRegistry::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        if (owner_ != nullptr) { owner_->release(index_, use_); }
        owner_ = std::exchange(other.owner_, nullptr);
        index_ = other.index_;
        use_   = other.use_;
    }
    return *this;
}

GenerationService& ModelRegistry::Lease::service() const {
    if (owner_ == nullptr) { throw std::logic_error("empty model lease"); }
    // The lease keeps the slot loaded, so its service lives as long as the lease does.
    return owner_->models_[index_].service->service();
}

const std::string& ModelRegistry::Lease::model() const {
    if (owner_ == nullptr) { throw std::logic_error("empty model lease"); }
    return owner_->models_[index_].definition.id;
}

} // namespace ninfer::serve
