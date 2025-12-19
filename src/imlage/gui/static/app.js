/**
 * IMLAGE GUI Application
 * Frontend JavaScript for image tagging interface
 */

// ============================================================================
// State
// ============================================================================

const state = {
    images: new Map(), // id -> { id, filename, thumbnail, width, height, format, results }
    selectedImage: null,
    settings: {
        plugins: [], // List of available plugins with their settings
        // Per-plugin settings: { name: { enabled, threshold, limit } }
        pluginSettings: {},
    },
    hardware: null,
    processing: false,
    abortProcessing: false,
};


// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    folderInput: document.getElementById('folderInput'),
    browseFilesBtn: document.getElementById('browseFilesBtn'),
    browseFolderBtn: document.getElementById('browseFolderBtn'),
    imageGrid: document.getElementById('imageGrid'),
    imageCount: document.getElementById('imageCount'),
    processingStatus: document.getElementById('processingStatus'),
    cancelBtn: document.getElementById('cancelBtn'),
    clearAllBtn: document.getElementById('clearAllBtn'),

    // Lightbox
    lightbox: document.getElementById('lightbox'),
    lightboxImage: document.getElementById('lightboxImage'),
    lightboxFilename: document.getElementById('lightboxFilename'),
    lightboxMetadata: document.getElementById('lightboxMetadata'),
    lightboxTags: document.getElementById('lightboxTags'),
    lightboxTagBtn: document.getElementById('lightboxTagBtn'),
    lightboxBack: document.getElementById('lightboxBack'),
    lightboxClose: document.getElementById('lightboxClose'),

    // Settings
    settingsBtn: document.getElementById('settingsBtn'),
    settingsModal: document.getElementById('settingsModal'),
    settingsClose: document.getElementById('settingsClose'),
    pluginsList: document.getElementById('pluginsList'),
    hardwareInfo: document.getElementById('hardwareInfo'),

    // Toast
    toastContainer: document.getElementById('toastContainer'),
};


// ============================================================================
// API Functions
// ============================================================================

async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        state.hardware = data.hardware;
        state.settings.plugins = data.plugins || [];

        // Initialize per-plugin settings with defaults
        for (const plugin of state.settings.plugins) {
            if (!state.settings.pluginSettings[plugin.name]) {
                state.settings.pluginSettings[plugin.name] = {
                    enabled: plugin.available, // Enable by default if available
                    threshold: 0.5,
                    limit: 50,
                    providesConfidence: plugin.provides_confidence !== false,
                };
            }
        }

        renderPluginsList();
        renderHardwareInfo();

        return data;
    } catch (error) {
        console.error('Failed to fetch status:', error);
        showToast('Failed to connect to server', 'error');
    }
}

async function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
}

async function tagImage(imageId) {
    // Get enabled plugins and their settings
    const enabledPlugins = [];
    const pluginConfigs = {};

    for (const plugin of state.settings.plugins) {
        const settings = state.settings.pluginSettings[plugin.name];
        if (settings && settings.enabled && plugin.available) {
            enabledPlugins.push(plugin.name);
            pluginConfigs[plugin.name] = {
                threshold: settings.threshold,
                limit: settings.limit,
            };
        }
    }

    if (enabledPlugins.length === 0) {
        throw new Error('No plugins enabled');
    }

    const response = await fetch(`/api/tag/${imageId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            plugins: enabledPlugins,
            plugin_configs: pluginConfigs,
        }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Tagging failed');
    }

    return response.json();
}

async function deleteImage(imageId) {
    await fetch(`/api/image/${imageId}`, { method: 'DELETE' });
}

async function clearAllImages() {
    await fetch('/api/images', { method: 'DELETE' });
}


// ============================================================================
// UI Rendering
// ============================================================================

function renderImageGrid() {
    elements.imageGrid.innerHTML = '';

    for (const [id, image] of state.images) {
        const card = createImageCard(image);
        elements.imageGrid.appendChild(card);
    }

    updateStatusBar();
}

function createImageCard(image) {
    const card = document.createElement('div');
    card.className = 'image-card';
    card.dataset.id = image.id;

    const hasResults = image.results && !image.results.error;
    const isProcessing = image.processing;

    card.innerHTML = `
        <img class="image-card-thumb" src="${image.thumbnail}" alt="${image.filename}">
        <div class="image-card-info">
            <div class="image-card-name">${image.filename}</div>
            <div class="image-card-meta">${image.width} x ${image.height}</div>
        </div>
        <div class="image-card-status ${hasResults ? 'tagged' : ''} ${isProcessing ? 'processing' : ''}">
            ${isProcessing ? `
                <svg class="spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="32"/>
                </svg>
            ` : hasResults ? `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
            ` : ''}
        </div>
    `;

    card.addEventListener('click', () => openLightbox(image.id));

    return card;
}

function updateStatusBar() {
    const count = state.images.size;
    elements.imageCount.textContent = `${count} image${count !== 1 ? 's' : ''}`;

    elements.cancelBtn.hidden = !state.processing;
    elements.clearAllBtn.disabled = count === 0 || state.processing;
}

function renderPluginsList() {
    if (!state.settings.plugins.length) {
        elements.pluginsList.innerHTML = '<p style="color: var(--text-tertiary); font-size: var(--font-size-sm);">No plugins found</p>';
        return;
    }

    elements.pluginsList.innerHTML = state.settings.plugins.map(plugin => {
        const settings = state.settings.pluginSettings[plugin.name] || { enabled: false, threshold: 0.5, limit: 50 };
        const displayName = plugin.display_name || plugin.name;
        const isAvailable = plugin.available;
        const isEnabled = settings.enabled && isAvailable;
        const providesConfidence = plugin.provides_confidence !== false;

        return `
            <div class="plugin-card" data-plugin="${plugin.name}">
                <div class="plugin-card-header">
                    <div class="plugin-card-title">
                        <span class="plugin-status-dot ${isAvailable ? 'available' : 'unavailable'}"></span>
                        <span class="plugin-name">${displayName}</span>
                    </div>
                    <label class="toggle ${!isAvailable ? 'disabled' : ''}">
                        <input type="checkbox"
                            ${isEnabled ? 'checked' : ''}
                            ${!isAvailable ? 'disabled' : ''}
                            onchange="togglePlugin('${plugin.name}', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                <div class="plugin-card-settings ${!isEnabled ? 'disabled' : ''}">
                    <div class="plugin-setting">
                        <label>Tag Limit</label>
                        <div class="setting-control">
                            <input type="range" min="5" max="100" value="${settings.limit}"
                                ${!isEnabled ? 'disabled' : ''}
                                oninput="updatePluginLimit('${plugin.name}', this.value)">
                            <span class="setting-value">${settings.limit}</span>
                        </div>
                    </div>
                    ${!providesConfidence ? `
                        <p class="plugin-note">Tags ordered by relevance (no confidence scores)</p>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');
}

function togglePlugin(pluginName, enabled) {
    if (state.settings.pluginSettings[pluginName]) {
        state.settings.pluginSettings[pluginName].enabled = enabled;
        renderPluginsList();
    }
}

function updatePluginThreshold(pluginName, value) {
    if (state.settings.pluginSettings[pluginName]) {
        state.settings.pluginSettings[pluginName].threshold = value / 100;
        // Update display value
        const card = document.querySelector(`[data-plugin="${pluginName}"]`);
        if (card) {
            const valueSpan = card.querySelector('.plugin-setting:first-child .setting-value');
            if (valueSpan) valueSpan.textContent = (value / 100).toFixed(2);
        }
    }
}

function updatePluginLimit(pluginName, value) {
    if (state.settings.pluginSettings[pluginName]) {
        state.settings.pluginSettings[pluginName].limit = parseInt(value);
        // Update display value
        const card = document.querySelector(`[data-plugin="${pluginName}"]`);
        if (card) {
            const valueSpan = card.querySelector('.plugin-setting:last-child .setting-value');
            if (valueSpan) valueSpan.textContent = value;
        }
    }
}

function renderHardwareInfo() {
    if (!state.hardware) {
        elements.hardwareInfo.innerHTML = '<p style="color: var(--text-tertiary);">Loading...</p>';
        return;
    }

    const hw = state.hardware;
    const gpuText = hw.gpu_type
        ? `${hw.gpu_name} (${hw.gpu_type.toUpperCase()})${hw.gpu_vram_gb ? ` - ${hw.gpu_vram_gb} GB` : ''}`
        : 'None detected';

    elements.hardwareInfo.innerHTML = `
        <div class="hardware-row">
            <span class="hardware-label">CPU</span>
            <span class="hardware-value">${hw.cpu}</span>
        </div>
        <div class="hardware-row">
            <span class="hardware-label">Cores</span>
            <span class="hardware-value">${hw.cpu_cores}</span>
        </div>
        <div class="hardware-row">
            <span class="hardware-label">RAM</span>
            <span class="hardware-value">${hw.ram_available_gb?.toFixed(1)} / ${hw.ram_total_gb?.toFixed(1)} GB</span>
        </div>
        <div class="hardware-row">
            <span class="hardware-label">GPU</span>
            <span class="hardware-value">${gpuText}</span>
        </div>
    `;
}


// ============================================================================
// Lightbox
// ============================================================================

async function openLightbox(imageId) {
    state.selectedImage = imageId;
    const image = state.images.get(imageId);

    if (!image) return;

    // Show lightbox immediately with thumbnail
    elements.lightbox.hidden = false;
    elements.lightboxFilename.textContent = image.filename;
    elements.lightboxImage.src = image.thumbnail;
    elements.lightboxMetadata.innerHTML = `
        <span>Size: ${image.width} x ${image.height}</span>
        <span>Format: ${image.format}</span>
    `;

    // Load full image (now served as a file)
    try {
        elements.lightboxImage.src = `/api/image/${imageId}`;

        // Get metadata including results
        const meta = await fetch(`/api/image/${imageId}/meta`).then(r => r.json());

        // Update results if available
        if (meta.results) {
            image.results = meta.results;
            state.images.set(imageId, image);
        }
    } catch (error) {
        console.error('Failed to load full image:', error);
    }

    renderLightboxTags(image);
}

function closeLightbox() {
    elements.lightbox.hidden = true;
    state.selectedImage = null;
}

/**
 * Get display name for a plugin (falls back to plugin name if not found)
 */
function getPluginDisplayName(pluginName) {
    const plugin = state.settings.plugins.find(p => p.name === pluginName);
    return plugin ? (plugin.display_name || plugin.name) : pluginName;
}

function renderLightboxTags(image) {
    if (!image.results || !image.results.results) {
        elements.lightboxTags.innerHTML = '<p class="no-tags">Click "Tag Image" to analyze</p>';
        elements.lightboxTagBtn.textContent = 'Tag Image';
        elements.lightboxTagBtn.disabled = false;
        return;
    }

    const results = image.results.results;
    let html = '';

    for (const [pluginName, pluginResult] of Object.entries(results)) {
        const displayName = getPluginDisplayName(pluginName);
        html += `<div class="plugin-result">`;

        if (pluginResult.error) {
            html += `
                <div class="plugin-header">
                    <span class="plugin-name">${displayName}</span>
                </div>
                <p class="plugin-error">${pluginResult.error}</p>
            `;
        } else {
            const tags = pluginResult.tags || [];
            const timeMs = pluginResult.inference_time_ms || 0;

            html += `
                <div class="plugin-header">
                    <span class="plugin-name">${displayName}</span>
                    <span class="plugin-time">${timeMs.toFixed(0)}ms</span>
                </div>
                <div class="tags-list">
                    ${tags.map(tag => {
                        // Only show confidence if provided
                        const hasConfidence = tag.confidence !== undefined && tag.confidence !== null;
                        if (hasConfidence) {
                            const conf = tag.confidence;
                            const confClass = conf >= 0.8 ? 'high-confidence' : conf >= 0.6 ? 'medium-confidence' : '';
                            return `
                                <span class="tag ${confClass}">
                                    ${tag.label}
                                    <span class="tag-confidence">${conf.toFixed(2)}</span>
                                </span>
                            `;
                        } else {
                            return `<span class="tag">${tag.label}</span>`;
                        }
                    }).join('')}
                </div>
            `;
        }

        html += '</div>';
    }

    elements.lightboxTags.innerHTML = html || '<p class="no-tags">No tags found</p>';
    elements.lightboxTagBtn.textContent = 'Re-tag Image';
    elements.lightboxTagBtn.disabled = false;
}


// ============================================================================
// Image Handling
// ============================================================================

/**
 * Recursively get files from DataTransferItemList (supports dropped folders)
 */
async function getFilesFromDataTransfer(items) {
    const files = [];
    const entries = [];

    // Get all entries first
    for (const item of items) {
        if (item.webkitGetAsEntry) {
            const entry = item.webkitGetAsEntry();
            if (entry) entries.push(entry);
        } else if (item.getAsFile) {
            const file = item.getAsFile();
            if (file) files.push(file);
        }
    }

    // Process entries (may include directories)
    async function processEntry(entry) {
        if (entry.isFile) {
            return new Promise((resolve) => {
                entry.file((file) => resolve([file]), () => resolve([]));
            });
        } else if (entry.isDirectory) {
            const dirReader = entry.createReader();
            const dirFiles = [];

            // Read directory entries in batches (readEntries may not return all at once)
            const readBatch = () => new Promise((resolve) => {
                dirReader.readEntries(async (entries) => {
                    if (entries.length === 0) {
                        resolve();
                    } else {
                        for (const e of entries) {
                            const subFiles = await processEntry(e);
                            dirFiles.push(...subFiles);
                        }
                        await readBatch(); // Continue reading
                        resolve();
                    }
                }, () => resolve());
            });

            await readBatch();
            return dirFiles;
        }
        return [];
    }

    // Process all entries
    for (const entry of entries) {
        const entryFiles = await processEntry(entry);
        files.push(...entryFiles);
    }

    return files;
}

async function handleFiles(files) {
    // Filter for image files - check both MIME type and extension
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif'];
    const imageFiles = Array.from(files).filter(f => {
        // Check MIME type
        if (f.type.startsWith('image/')) return true;
        // Fallback to extension check (for folder selections where MIME may not be set)
        const ext = f.name.toLowerCase().substring(f.name.lastIndexOf('.'));
        return imageExtensions.includes(ext);
    });

    if (imageFiles.length === 0) {
        showToast('No valid image files found', 'error');
        return;
    }

    // Upload all images first
    const uploadedIds = [];
    for (const file of imageFiles) {
        try {
            const result = await uploadImage(file);
            state.images.set(result.id, {
                ...result,
                processing: false,
                results: null,
            });
            uploadedIds.push(result.id);
            renderImageGrid(); // Update grid as each image uploads
        } catch (error) {
            showToast(`Failed to upload ${file.name}: ${error.message}`, 'error');
        }
    }

    showToast(`Processing ${uploadedIds.length} image${uploadedIds.length !== 1 ? 's' : ''}...`);

    // Auto-start tagging queue for uploaded images
    if (uploadedIds.length > 0 && !state.processing) {
        autoTagImages(uploadedIds);
    }
}

async function autoTagImages(imageIds) {
    if (state.processing) return;

    state.processing = true;
    state.abortProcessing = false;
    updateStatusBar();

    let completed = 0;
    for (const id of imageIds) {
        if (state.abortProcessing) {
            break;
        }

        elements.processingStatus.textContent = `Tagging ${completed + 1} of ${imageIds.length}...`;

        try {
            await tagSingleImage(id);
        } catch (error) {
            console.error(`Failed to tag image ${id}:`, error);
        }

        completed++;
    }

    const wasCancelled = state.abortProcessing;
    state.processing = false;
    state.abortProcessing = false;
    elements.processingStatus.textContent = '';
    updateStatusBar();

    if (wasCancelled) {
        showToast(`Cancelled after ${completed} image${completed !== 1 ? 's' : ''}`, 'info');
    } else {
        showToast(`Tagged ${completed} image${completed !== 1 ? 's' : ''}`, 'success');
    }
}

async function tagSingleImage(imageId) {
    const image = state.images.get(imageId);
    if (!image) return;

    image.processing = true;
    state.images.set(imageId, image);

    if (state.selectedImage === imageId) {
        elements.lightboxTagBtn.disabled = true;
        elements.lightboxTagBtn.textContent = 'Processing...';
    }

    renderImageGrid();

    try {
        const result = await tagImage(imageId);
        image.results = result;
        image.processing = false;
        state.images.set(imageId, image);

        if (state.selectedImage === imageId) {
            renderLightboxTags(image);
        }

        renderImageGrid();
    } catch (error) {
        image.processing = false;
        state.images.set(imageId, image);
        showToast(`Tagging failed: ${error.message}`, 'error');

        if (state.selectedImage === imageId) {
            elements.lightboxTagBtn.disabled = false;
            elements.lightboxTagBtn.textContent = 'Tag Image';
        }

        renderImageGrid();
    }
}

function cancelProcessing() {
    if (state.processing) {
        state.abortProcessing = true;
    }
}

async function clearAll() {
    if (!confirm('Clear all images?')) return;

    try {
        await clearAllImages();
        state.images.clear();
        renderImageGrid();
        closeLightbox();
        showToast('All images cleared');
    } catch (error) {
        showToast('Failed to clear images', 'error');
    }
}


// ============================================================================
// Settings
// ============================================================================

function openSettings() {
    elements.settingsModal.hidden = false;
}

function closeSettings() {
    elements.settingsModal.hidden = true;
}


// ============================================================================
// Toast Notifications
// ============================================================================

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}


// ============================================================================
// Event Listeners
// ============================================================================

// Drop zone - browse buttons
elements.browseFilesBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    elements.fileInput.click();
});
elements.browseFolderBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    elements.folderInput.click();
});
elements.fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
elements.folderInput.addEventListener('change', (e) => handleFiles(e.target.files));

elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('drag-over');
});

elements.dropZone.addEventListener('dragleave', () => {
    elements.dropZone.classList.remove('drag-over');
});

elements.dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-over');

    // Handle dropped items - may include folders
    const items = e.dataTransfer.items;
    if (items && items.length > 0) {
        const files = await getFilesFromDataTransfer(items);
        if (files.length > 0) {
            handleFiles(files);
        } else {
            showToast('No valid image files found', 'error');
        }
    } else {
        handleFiles(e.dataTransfer.files);
    }
});

// Global drag and drop
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', async (e) => {
    e.preventDefault();
    if (e.target !== elements.dropZone && !elements.dropZone.contains(e.target)) {
        const items = e.dataTransfer.items;
        if (items && items.length > 0) {
            const files = await getFilesFromDataTransfer(items);
            if (files.length > 0) {
                handleFiles(files);
            }
        } else {
            handleFiles(e.dataTransfer.files);
        }
    }
});

// Action buttons
elements.cancelBtn.addEventListener('click', cancelProcessing);
elements.clearAllBtn.addEventListener('click', clearAll);

// Lightbox
elements.lightboxBack.addEventListener('click', closeLightbox);
elements.lightboxClose.addEventListener('click', closeLightbox);
elements.lightbox.querySelector('.lightbox-overlay').addEventListener('click', closeLightbox);
elements.lightboxTagBtn.addEventListener('click', () => {
    if (state.selectedImage) {
        tagSingleImage(state.selectedImage);
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (!elements.lightbox.hidden) {
            closeLightbox();
        } else if (!elements.settingsModal.hidden) {
            closeSettings();
        }
    }
});

// Settings
elements.settingsBtn.addEventListener('click', openSettings);
elements.settingsClose.addEventListener('click', closeSettings);
elements.settingsModal.querySelector('.modal-overlay').addEventListener('click', closeSettings);


// ============================================================================
// Initialize
// ============================================================================

async function init() {
    // Fetch system status (initializes per-plugin settings)
    await fetchStatus();

    // Initial render
    renderImageGrid();
}

// Start app
init();
