/**
 * Visual Buffet GUI Application
 * Frontend JavaScript for image tagging interface
 */

// ============================================================================
// State
// ============================================================================

const state = {
    images: new Map(), // id -> { id, filename, thumbnail, width, height, format, results }
    selectedImageId: null, // Currently selected card (for panel)
    panelOpen: false,      // Is detail panel visible
    settings: {
        plugins: [], // List of available plugins with their settings
        // Per-plugin settings: { name: { enabled, threshold, limit, discoveryMode, useRamPlus, useFlorence2 } }
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

    // Detail Panel
    detailPanel: document.getElementById('detailPanel'),
    panelBack: document.getElementById('panelBack'),
    panelClose: document.getElementById('panelClose'),
    panelTitle: document.getElementById('panelTitle'),
    panelPreview: document.getElementById('panelPreview'),
    panelPreviewImg: document.getElementById('panelPreviewImg'),
    panelMetadata: document.getElementById('panelMetadata'),
    panelTags: document.getElementById('panelTags'),
    panelViewBtn: document.getElementById('panelViewBtn'),
    panelTagBtn: document.getElementById('panelTagBtn'),

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
    pluginsSection: document.getElementById('pluginsSection'),
    hardwareInfo: document.getElementById('hardwareInfo'),
    qualityButtons: document.querySelectorAll('.quality-btn'),
    customQualityBtn: document.getElementById('customQualityBtn'),
    qualityHint: document.getElementById('qualityHint'),

    // Header
    appVersion: document.getElementById('appVersion'),

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

        // Update version in header
        if (data.version && elements.appVersion) {
            elements.appVersion.textContent = `v${data.version}`;
        }

        state.hardware = data.hardware;
        state.settings.plugins = data.plugins || [];

        // Load saved settings from server
        let savedSettings = {};
        try {
            const settingsResponse = await fetch('/api/settings');
            const settingsData = await settingsResponse.json();
            savedSettings = settingsData.plugin_settings || {};
        } catch (e) {
            console.warn('Could not load saved settings:', e);
        }

        // Initialize per-plugin settings - use saved values or defaults
        for (const plugin of state.settings.plugins) {
            if (!state.settings.pluginSettings[plugin.name]) {
                const saved = savedSettings[plugin.name];
                if (saved) {
                    // Use saved settings
                    state.settings.pluginSettings[plugin.name] = {
                        enabled: saved.enabled ?? plugin.available,
                        threshold: saved.threshold ?? plugin.recommended_threshold ?? 0.0,
                        limit: saved.limit ?? 50,
                        quality: saved.quality ?? 'standard',
                        providesConfidence: plugin.provides_confidence !== false,
                        // SigLIP discovery settings
                        discoveryMode: saved.discoveryMode ?? false,
                        useRamPlus: saved.useRamPlus ?? true,
                        useFlorence2: saved.useFlorence2 ?? true,
                    };
                } else {
                    // Use plugin's recommended_threshold (critical for SigLIP which needs 0.01)
                    const threshold = plugin.recommended_threshold ?? 0.0;
                    state.settings.pluginSettings[plugin.name] = {
                        enabled: plugin.available,
                        threshold: threshold,
                        limit: 50,
                        quality: 'standard',
                        providesConfidence: plugin.provides_confidence !== false,
                        // SigLIP discovery settings (defaults)
                        discoveryMode: false,
                        useRamPlus: true,
                        useFlorence2: true,
                    };
                }
            }
        }

        renderPluginsList();
        renderHardwareInfo();
        updateGlobalQualityUI();

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
            const config = {
                threshold: settings.threshold,
                limit: settings.limit,
                quality: settings.quality || 'standard',
            };
            // Add discovery settings for SigLIP
            if (plugin.name === 'siglip') {
                config.discovery_mode = settings.discoveryMode || false;
                config.use_ram_plus = settings.useRamPlus !== false;
                config.use_florence_2 = settings.useFlorence2 !== false;
            }
            pluginConfigs[plugin.name] = config;
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

async function saveSettings() {
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                plugin_settings: state.settings.pluginSettings,
            }),
        });
    } catch (error) {
        console.error('Failed to save settings:', error);
    }
}

async function clearAllImages() {
    await fetch('/api/images', { method: 'DELETE' });
}

async function fetchExistingImages() {
    try {
        const response = await fetch('/api/images');
        const data = await response.json();

        for (const image of data.images || []) {
            state.images.set(image.id, {
                id: image.id,
                filename: image.filename,
                width: image.width,
                height: image.height,
                format: image.format,
                thumbnail: image.thumbnail,
                processing: false,
                results: image.results ? { results: image.results } : null,
            });
        }

        renderImageGrid();
    } catch (error) {
        console.error('Failed to fetch existing images:', error);
    }
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
    card.tabIndex = 0; // Make focusable for keyboard nav

    const hasResults = image.results && !image.results.error;
    const isProcessing = image.processing;
    const isSelected = state.selectedImageIdId === image.id;

    // Add selected class if this card is selected
    if (isSelected) {
        card.classList.add('selected');
    }

    // Count tags for display
    let tagCount = 0;
    if (hasResults && image.results.results) {
        for (const pluginResult of Object.values(image.results.results)) {
            if (pluginResult.tags) {
                tagCount += pluginResult.tags.length;
            }
        }
    }

    // Thumbnail image
    const thumb = document.createElement('img');
    thumb.className = 'image-card-thumb';
    thumb.src = image.thumbnail;
    thumb.alt = image.filename;
    card.appendChild(thumb);

    // Info section
    const info = document.createElement('div');
    info.className = 'image-card-info';

    const name = document.createElement('div');
    name.className = 'image-card-name';
    name.textContent = image.filename;
    info.appendChild(name);

    const meta = document.createElement('div');
    meta.className = 'image-card-meta';
    meta.textContent = hasResults ? `✓ ${tagCount} tags` : `${image.width} × ${image.height}`;
    info.appendChild(meta);

    card.appendChild(info);

    // Status indicator
    const status = document.createElement('div');
    status.className = 'image-card-status';
    if (hasResults) status.classList.add('tagged');
    if (isProcessing) status.classList.add('processing');

    if (isProcessing) {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'spinner');
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('fill', 'none');
        svg.setAttribute('stroke', 'currentColor');
        svg.setAttribute('stroke-width', '2');
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', '12');
        circle.setAttribute('cy', '12');
        circle.setAttribute('r', '10');
        circle.setAttribute('stroke-dasharray', '32');
        circle.setAttribute('stroke-dashoffset', '32');
        svg.appendChild(circle);
        status.appendChild(svg);
    } else if (hasResults) {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('fill', 'none');
        svg.setAttribute('stroke', 'currentColor');
        svg.setAttribute('stroke-width', '2');
        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.setAttribute('points', '20 6 9 17 4 12');
        svg.appendChild(polyline);
        status.appendChild(svg);
    }

    card.appendChild(status);

    // Click opens panel (not lightbox)
    card.addEventListener('click', () => openPanel(image.id));

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
        elements.pluginsList.textContent = 'No plugins found';
        return;
    }

    // Clear existing content using safe DOM methods
    elements.pluginsList.textContent = '';

    // Sort plugins: taggers first, then detection models
    const pluginOrder = ['florence_2', 'ram_plus', 'siglip', 'yolo'];
    const sortedPlugins = [...state.settings.plugins].sort((a, b) => {
        const aIndex = pluginOrder.indexOf(a.name);
        const bIndex = pluginOrder.indexOf(b.name);
        // Unknown plugins go at the end
        const aOrder = aIndex === -1 ? 999 : aIndex;
        const bOrder = bIndex === -1 ? 999 : bIndex;
        return aOrder - bOrder;
    });

    for (const plugin of sortedPlugins) {
        const card = createPluginCard(plugin);
        elements.pluginsList.appendChild(card);
    }
}

function createPluginCard(plugin) {
    const defaultThreshold = plugin.recommended_threshold ?? 0.0;
    const settings = state.settings.pluginSettings[plugin.name] || { enabled: false, threshold: defaultThreshold, limit: 50 };
    const displayName = plugin.display_name || plugin.name;
    const isAvailable = plugin.available;
    const isEnabled = settings.enabled && isAvailable;
    const isSigLIP = plugin.name === 'siglip';
    const discoveryEnabled = settings.discoveryMode || false;

    const card = document.createElement('div');
    card.className = 'plugin-card';
    card.dataset.plugin = plugin.name;

    // Header
    const header = document.createElement('div');
    header.className = 'plugin-card-header';

    const title = document.createElement('div');
    title.className = 'plugin-card-title';

    const statusDot = document.createElement('span');
    statusDot.className = `plugin-status-dot ${isAvailable ? 'available' : 'unavailable'}`;
    title.appendChild(statusDot);

    const nameSpan = document.createElement('span');
    nameSpan.className = 'plugin-name';
    nameSpan.textContent = displayName;
    title.appendChild(nameSpan);

    header.appendChild(title);

    // Toggle
    const toggleLabel = document.createElement('label');
    toggleLabel.className = `toggle ${!isAvailable ? 'disabled' : ''}`;
    const toggleInput = document.createElement('input');
    toggleInput.type = 'checkbox';
    toggleInput.checked = isEnabled;
    toggleInput.disabled = !isAvailable;
    toggleInput.addEventListener('change', () => togglePlugin(plugin.name, toggleInput.checked));
    const toggleSlider = document.createElement('span');
    toggleSlider.className = 'toggle-slider';
    toggleLabel.appendChild(toggleInput);
    toggleLabel.appendChild(toggleSlider);
    header.appendChild(toggleLabel);

    card.appendChild(header);

    // Settings container
    const settingsDiv = document.createElement('div');
    settingsDiv.className = `plugin-card-settings ${!isEnabled ? 'disabled' : ''}`;

    // SigLIP Discovery Mode settings
    if (isSigLIP) {
        // Discovery toggle
        const discoverySetting = document.createElement('div');
        discoverySetting.className = 'plugin-setting discovery-toggle';

        const discoveryLabel = document.createElement('label');
        discoveryLabel.textContent = 'Discovery Mode';
        discoverySetting.appendChild(discoveryLabel);

        const discoveryControl = document.createElement('div');
        discoveryControl.className = 'setting-control';

        const discoveryToggleLabel = document.createElement('label');
        discoveryToggleLabel.className = `toggle ${!isEnabled ? 'disabled' : ''}`;
        const discoveryToggleInput = document.createElement('input');
        discoveryToggleInput.type = 'checkbox';
        discoveryToggleInput.checked = discoveryEnabled;
        discoveryToggleInput.disabled = !isEnabled;
        discoveryToggleInput.addEventListener('change', () => toggleDiscoveryMode(plugin.name, discoveryToggleInput.checked));
        const discoveryToggleSlider = document.createElement('span');
        discoveryToggleSlider.className = 'toggle-slider';
        discoveryToggleLabel.appendChild(discoveryToggleInput);
        discoveryToggleLabel.appendChild(discoveryToggleSlider);
        discoveryControl.appendChild(discoveryToggleLabel);

        const discoveryHint = document.createElement('span');
        discoveryHint.className = 'setting-hint';
        discoveryHint.textContent = 'Use RAM++/Florence-2 for vocabulary';
        discoveryControl.appendChild(discoveryHint);

        discoverySetting.appendChild(discoveryControl);
        settingsDiv.appendChild(discoverySetting);

        // Discovery sources
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = `discovery-sources ${!discoveryEnabled || !isEnabled ? 'disabled' : ''}`;

        const ramLabel = document.createElement('label');
        ramLabel.className = 'checkbox-label';
        const ramCheck = document.createElement('input');
        ramCheck.type = 'checkbox';
        ramCheck.checked = settings.useRamPlus !== false;
        ramCheck.disabled = !discoveryEnabled || !isEnabled;
        ramCheck.addEventListener('change', () => updateDiscoverySource(plugin.name, 'useRamPlus', ramCheck.checked));
        ramLabel.appendChild(ramCheck);
        const ramText = document.createElement('span');
        ramText.textContent = 'RAM++';
        ramLabel.appendChild(ramText);
        sourcesDiv.appendChild(ramLabel);

        const florenceLabel = document.createElement('label');
        florenceLabel.className = 'checkbox-label';
        const florenceCheck = document.createElement('input');
        florenceCheck.type = 'checkbox';
        florenceCheck.checked = settings.useFlorence2 !== false;
        florenceCheck.disabled = !discoveryEnabled || !isEnabled;
        florenceCheck.addEventListener('change', () => updateDiscoverySource(plugin.name, 'useFlorence2', florenceCheck.checked));
        florenceLabel.appendChild(florenceCheck);
        const florenceText = document.createElement('span');
        florenceText.textContent = 'Florence-2';
        florenceLabel.appendChild(florenceText);
        sourcesDiv.appendChild(florenceLabel);

        settingsDiv.appendChild(sourcesDiv);
    }

    // Threshold setting (only for plugins with confidence)
    if (plugin.provides_confidence) {
        const thresholdSetting = document.createElement('div');
        thresholdSetting.className = 'plugin-setting';

        const thresholdLabel = document.createElement('label');
        thresholdLabel.textContent = 'Threshold';
        thresholdSetting.appendChild(thresholdLabel);

        const thresholdControl = document.createElement('div');
        thresholdControl.className = 'setting-control';

        const thresholdRange = document.createElement('input');
        thresholdRange.type = 'range';
        thresholdRange.min = '0';
        thresholdRange.max = '100';
        thresholdRange.value = Math.round(settings.threshold * 100);
        thresholdRange.disabled = !isEnabled;
        const thresholdValue = document.createElement('span');
        thresholdValue.className = 'setting-value';
        thresholdValue.textContent = settings.threshold.toFixed(2);
        thresholdRange.addEventListener('input', () => {
            updatePluginThreshold(plugin.name, thresholdRange.value);
            thresholdValue.textContent = (thresholdRange.value / 100).toFixed(2);
        });
        thresholdControl.appendChild(thresholdRange);
        thresholdControl.appendChild(thresholdValue);

        thresholdSetting.appendChild(thresholdControl);
        settingsDiv.appendChild(thresholdSetting);
    }

    // Quality setting
    const qualitySetting = document.createElement('div');
    qualitySetting.className = 'plugin-setting';

    const qualityLabel = document.createElement('label');
    qualityLabel.textContent = 'Quality';
    qualitySetting.appendChild(qualityLabel);

    const qualityControl = document.createElement('div');
    qualityControl.className = 'setting-control';

    const qualitySelect = document.createElement('select');
    qualitySelect.className = 'quality-select';
    qualitySelect.disabled = !isEnabled;
    ['quick', 'standard', 'max'].forEach(q => {
        const opt = document.createElement('option');
        opt.value = q;
        opt.textContent = q.charAt(0).toUpperCase() + q.slice(1);
        opt.selected = settings.quality === q;
        qualitySelect.appendChild(opt);
    });
    qualitySelect.addEventListener('change', () => updatePluginQuality(plugin.name, qualitySelect.value));
    qualityControl.appendChild(qualitySelect);

    qualitySetting.appendChild(qualityControl);
    settingsDiv.appendChild(qualitySetting);

    // Limit setting
    const limitSetting = document.createElement('div');
    limitSetting.className = 'plugin-setting';

    const limitLabel = document.createElement('label');
    limitLabel.textContent = 'Tag Limit';
    limitSetting.appendChild(limitLabel);

    const limitControl = document.createElement('div');
    limitControl.className = 'setting-control';

    const limitRange = document.createElement('input');
    limitRange.type = 'range';
    limitRange.min = '5';
    limitRange.max = '100';
    limitRange.value = settings.limit;
    limitRange.disabled = !isEnabled;
    const limitValue = document.createElement('span');
    limitValue.className = 'setting-value';
    limitValue.textContent = settings.limit;
    limitRange.addEventListener('input', () => {
        updatePluginLimit(plugin.name, limitRange.value);
        limitValue.textContent = limitRange.value;
    });
    limitControl.appendChild(limitRange);
    limitControl.appendChild(limitValue);

    limitSetting.appendChild(limitControl);
    settingsDiv.appendChild(limitSetting);

    card.appendChild(settingsDiv);
    return card;
}

function togglePlugin(pluginName, enabled) {
    if (state.settings.pluginSettings[pluginName]) {
        state.settings.pluginSettings[pluginName].enabled = enabled;
        renderPluginsList();
        saveSettings(); // Persist to disk
    }
}

function updatePluginThreshold(pluginName, value) {
    if (state.settings.pluginSettings[pluginName]) {
        state.settings.pluginSettings[pluginName].threshold = value / 100;
        // Update display value - find the threshold setting specifically
        const card = document.querySelector(`[data-plugin="${pluginName}"]`);
        if (card) {
            const thresholdSetting = Array.from(card.querySelectorAll('.plugin-setting'))
                .find(el => el.querySelector('label')?.textContent === 'Threshold');
            if (thresholdSetting) {
                const valueSpan = thresholdSetting.querySelector('.setting-value');
                if (valueSpan) valueSpan.textContent = (value / 100).toFixed(2);
            }
        }
        saveSettings(); // Persist to disk
    }
}

function updatePluginLimit(pluginName, value) {
    if (state.settings.pluginSettings[pluginName]) {
        state.settings.pluginSettings[pluginName].limit = parseInt(value);
        // Update display value - find the limit setting specifically
        const card = document.querySelector(`[data-plugin="${pluginName}"]`);
        if (card) {
            const limitSetting = Array.from(card.querySelectorAll('.plugin-setting'))
                .find(el => el.querySelector('label')?.textContent === 'Tag Limit');
            if (limitSetting) {
                const valueSpan = limitSetting.querySelector('.setting-value');
                if (valueSpan) valueSpan.textContent = value;
            }
        }
        saveSettings(); // Persist to disk
    }
}

function updatePluginQuality(pluginName, value) {
    if (state.settings.pluginSettings[pluginName]) {
        state.settings.pluginSettings[pluginName].quality = value;
        updateGlobalQualityUI(); // Sync global quality buttons
        saveSettings(); // Persist to disk
    }
}

function toggleDiscoveryMode(pluginName, enabled) {
    if (state.settings.pluginSettings[pluginName]) {
        state.settings.pluginSettings[pluginName].discoveryMode = enabled;
        renderPluginsList();
        saveSettings();
    }
}

function updateDiscoverySource(pluginName, source, enabled) {
    if (state.settings.pluginSettings[pluginName]) {
        state.settings.pluginSettings[pluginName][source] = enabled;
        saveSettings();
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
// Detail Panel
// ============================================================================

/**
 * Open the detail panel for an image
 */
async function openPanel(imageId) {
    const image = state.images.get(imageId);
    if (!image) return;

    // Update state
    state.selectedImageIdId = imageId;
    state.panelOpen = true;

    // Update UI
    elements.detailPanel.classList.add('open');
    elements.detailPanel.setAttribute('aria-hidden', 'false');
    elements.imageGrid.classList.add('panel-open');

    // Update panel content
    elements.panelTitle.textContent = image.filename;
    elements.panelPreviewImg.src = image.thumbnail;
    elements.panelPreviewImg.alt = image.filename;

    // Update metadata
    elements.panelMetadata.textContent = '';
    const sizeSpan = document.createElement('span');
    sizeSpan.textContent = `${image.width} × ${image.height}`;
    elements.panelMetadata.appendChild(sizeSpan);

    const sep = document.createElement('span');
    sep.textContent = '•';
    elements.panelMetadata.appendChild(sep);

    const formatSpan = document.createElement('span');
    formatSpan.textContent = image.format;
    elements.panelMetadata.appendChild(formatSpan);

    // Render grid to update selected state
    renderImageGrid();

    // Render tags in panel
    renderPanelTags(image);

    // Try to load full preview and any server-side results
    try {
        elements.panelPreviewImg.src = `/api/image/${imageId}`;

        const meta = await fetch(`/api/image/${imageId}/meta`).then(r => r.json());
        if (meta.results) {
            image.results = meta.results;
            state.images.set(imageId, image);
            renderPanelTags(image);
        }
    } catch (error) {
        console.error('Failed to load image details:', error);
    }

    // Move focus to close button for accessibility
    requestAnimationFrame(() => {
        elements.panelClose.focus();
    });
}

/**
 * Close the detail panel
 */
function closePanel() {
    const previouslySelected = state.selectedImageIdId;

    state.panelOpen = false;
    state.selectedImageIdId = null;

    elements.detailPanel.classList.remove('open');
    elements.detailPanel.setAttribute('aria-hidden', 'true');
    elements.imageGrid.classList.remove('panel-open');

    // Re-render grid to remove selected state
    renderImageGrid();

    // Return focus to the card that was selected
    if (previouslySelected) {
        requestAnimationFrame(() => {
            const card = document.querySelector(`[data-id="${previouslySelected}"]`);
            if (card) card.focus();
        });
    }
}

/**
 * Render tag results in the panel with the new row format
 */
function renderPanelTags(image) {
    // Clear existing content
    elements.panelTags.textContent = '';

    if (!image.results || !image.results.results) {
        const emptyState = document.createElement('p');
        emptyState.className = 'panel-empty-state';
        emptyState.textContent = 'Click "Tag Image" to analyze';
        elements.panelTags.appendChild(emptyState);
        elements.panelTagBtn.textContent = 'Tag Image';
        elements.panelTagBtn.disabled = false;
        return;
    }

    const results = image.results.results;
    let hasAnyTags = false;

    for (const [pluginName, pluginResult] of Object.entries(results)) {
        const displayName = getPluginDisplayName(pluginName);
        const section = document.createElement('div');
        section.className = 'panel-plugin-section';

        // Plugin header
        const header = document.createElement('div');
        header.className = 'panel-plugin-header';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'panel-plugin-name';
        nameSpan.textContent = displayName;
        header.appendChild(nameSpan);

        if (pluginResult.error) {
            const errorSpan = document.createElement('span');
            errorSpan.className = 'panel-plugin-error';
            errorSpan.textContent = 'Error';
            header.appendChild(errorSpan);
            section.appendChild(header);

            const errorMsg = document.createElement('p');
            errorMsg.className = 'panel-empty-state';
            errorMsg.textContent = pluginResult.error;
            section.appendChild(errorMsg);
        } else {
            const tags = pluginResult.tags || [];
            hasAnyTags = hasAnyTags || tags.length > 0;

            const countSpan = document.createElement('span');
            countSpan.className = 'panel-plugin-count';
            countSpan.textContent = `${tags.length} tags`;
            header.appendChild(countSpan);

            if (pluginResult.inference_time_ms) {
                const timeSpan = document.createElement('span');
                timeSpan.className = 'panel-plugin-time';
                timeSpan.textContent = `${pluginResult.inference_time_ms.toFixed(0)}ms`;
                header.appendChild(timeSpan);
            }

            section.appendChild(header);

            // Tag rows
            const tagList = document.createElement('div');
            tagList.className = 'panel-tag-list';

            for (const tag of tags) {
                const row = document.createElement('div');
                row.className = 'panel-tag-row';

                // Label
                const label = document.createElement('span');
                label.className = 'panel-tag-label';
                label.textContent = tag.label;
                row.appendChild(label);

                // Confidence
                if (tag.confidence !== undefined && tag.confidence !== null) {
                    const conf = tag.confidence;
                    const confSpan = document.createElement('span');
                    confSpan.className = 'panel-tag-confidence';
                    if (conf >= 0.8) confSpan.classList.add('high');
                    else if (conf >= 0.6) confSpan.classList.add('medium');
                    else confSpan.classList.add('low');
                    confSpan.textContent = `${Math.round(conf * 100)}%`;
                    row.appendChild(confSpan);
                }

                // Source dots (for multi-resolution results)
                if (tag.sources !== undefined && tag.max_sources !== undefined) {
                    const dotsDiv = document.createElement('div');
                    dotsDiv.className = 'panel-tag-dots';

                    for (let i = 0; i < tag.max_sources; i++) {
                        const dot = document.createElement('span');
                        dot.className = 'panel-tag-dot';
                        if (i < tag.sources) dot.classList.add('filled');
                        dotsDiv.appendChild(dot);
                    }
                    row.appendChild(dotsDiv);

                    const sourcesSpan = document.createElement('span');
                    sourcesSpan.className = 'panel-tag-sources';
                    sourcesSpan.textContent = `${tag.sources}/${tag.max_sources}`;
                    row.appendChild(sourcesSpan);
                }

                tagList.appendChild(row);
            }

            section.appendChild(tagList);
        }

        elements.panelTags.appendChild(section);
    }

    if (!hasAnyTags) {
        const emptyState = document.createElement('p');
        emptyState.className = 'panel-empty-state';
        emptyState.textContent = 'No tags found';
        elements.panelTags.appendChild(emptyState);
    }

    elements.panelTagBtn.textContent = 'Re-tag';
    elements.panelTagBtn.disabled = false;
}


// ============================================================================
// Lightbox
// ============================================================================

async function openLightbox(imageId) {
    state.selectedImageIdId = imageId;
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
    state.selectedImageId = null;
}

/**
 * Get display name for a plugin (falls back to plugin name if not found)
 */
function getPluginDisplayName(pluginName) {
    const plugin = state.settings.plugins.find(p => p.name === pluginName);
    return plugin ? (plugin.display_name || plugin.name) : pluginName;
}

function renderLightboxTags(image) {
    // Clear existing content
    elements.lightboxTags.textContent = '';

    if (!image.results) {
        const noTags = document.createElement('p');
        noTags.className = 'no-tags';
        noTags.textContent = 'Click "Tag Image" to analyze';
        elements.lightboxTags.appendChild(noTags);
        elements.lightboxTagBtn.textContent = 'Tag Image';
        elements.lightboxTagBtn.disabled = false;
        return;
    }

    // Standard results
    if (!image.results.results) {
        const noTags = document.createElement('p');
        noTags.className = 'no-tags';
        noTags.textContent = 'Click "Tag Image" to analyze';
        elements.lightboxTags.appendChild(noTags);
        elements.lightboxTagBtn.textContent = 'Tag Image';
        elements.lightboxTagBtn.disabled = false;
        return;
    }

    // Check if SigLIP result has discovery mode enabled (check metadata)
    const siglipResult = image.results.results.siglip;
    if (siglipResult && siglipResult.metadata && siglipResult.metadata.discovery_mode) {
        renderDiscoveryTags(image);
        return;
    }

    const results = image.results.results;

    for (const [pluginName, pluginResult] of Object.entries(results)) {
        const displayName = getPluginDisplayName(pluginName);
        const pluginDiv = document.createElement('div');
        pluginDiv.className = 'plugin-result';

        if (pluginResult.error) {
            const header = document.createElement('div');
            header.className = 'plugin-header';
            const nameSpan = document.createElement('span');
            nameSpan.className = 'plugin-name';
            nameSpan.textContent = displayName;
            header.appendChild(nameSpan);

            const errorP = document.createElement('p');
            errorP.className = 'plugin-error';
            errorP.textContent = pluginResult.error;

            pluginDiv.appendChild(header);
            pluginDiv.appendChild(errorP);
        } else {
            const tags = pluginResult.tags || [];
            const timeMs = pluginResult.inference_time_ms || 0;

            const header = document.createElement('div');
            header.className = 'plugin-header';
            const nameSpan = document.createElement('span');
            nameSpan.className = 'plugin-name';
            nameSpan.textContent = displayName;
            const timeSpan = document.createElement('span');
            timeSpan.className = 'plugin-time';
            timeSpan.textContent = `${timeMs.toFixed(0)}ms`;
            header.appendChild(nameSpan);
            header.appendChild(timeSpan);

            const tagsList = document.createElement('div');
            tagsList.className = 'tags-list';

            for (const tag of tags) {
                const tagSpan = document.createElement('span');
                const hasConfidence = tag.confidence !== undefined && tag.confidence !== null;

                if (hasConfidence) {
                    const conf = tag.confidence;
                    const confClass = conf >= 0.8 ? 'high-confidence' : conf >= 0.6 ? 'medium-confidence' : '';
                    tagSpan.className = `tag ${confClass}`;
                    tagSpan.textContent = tag.label + ' ';
                    const confSpan = document.createElement('span');
                    confSpan.className = 'tag-confidence';
                    confSpan.textContent = Math.round(conf * 100) + '%';
                    tagSpan.appendChild(confSpan);
                } else {
                    tagSpan.className = 'tag';
                    tagSpan.textContent = tag.label;
                }
                tagsList.appendChild(tagSpan);
            }

            pluginDiv.appendChild(header);
            pluginDiv.appendChild(tagsList);
        }

        elements.lightboxTags.appendChild(pluginDiv);
    }

    if (elements.lightboxTags.children.length === 0) {
        const noTags = document.createElement('p');
        noTags.className = 'no-tags';
        noTags.textContent = 'No tags found';
        elements.lightboxTags.appendChild(noTags);
    }

    elements.lightboxTagBtn.textContent = 'Re-tag Image';
    elements.lightboxTagBtn.disabled = false;
}

function renderDiscoveryTags(image) {
    const result = image.results;
    const siglipResult = result.results.siglip;
    const metadata = siglipResult.metadata || {};

    // Clear existing content
    elements.lightboxTags.textContent = '';

    // Discovery summary header
    const sources = metadata.discovery_sources || [];
    const vocabSize = metadata.vocabulary_size || 0;

    const summaryDiv = document.createElement('div');
    summaryDiv.className = 'pipeline-summary';

    const discoveryHeader = document.createElement('div');
    discoveryHeader.className = 'pipeline-header';
    const badge = document.createElement('span');
    badge.className = 'pipeline-badge';
    badge.textContent = 'Discovery Mode';
    discoveryHeader.appendChild(badge);

    const statsDiv = document.createElement('div');
    statsDiv.className = 'pipeline-stats';
    const statSpan = document.createElement('span');
    statSpan.className = 'stat';
    statSpan.textContent = `${vocabSize} candidates from ${sources.map(s => getPluginDisplayName(s)).join(' + ')}`;
    statsDiv.appendChild(statSpan);

    summaryDiv.appendChild(discoveryHeader);
    summaryDiv.appendChild(statsDiv);
    elements.lightboxTags.appendChild(summaryDiv);

    // Render SigLIP scored results first
    renderPluginResultDiv('siglip', siglipResult, 'scorer');

    // Render discovery plugin results from metadata
    const discoveryResults = metadata.discovery_results || {};
    for (const [pluginName, pluginResult] of Object.entries(discoveryResults)) {
        renderPluginResultDiv(pluginName, pluginResult, 'discovery');
    }

    // Render any other plugins that ran (non-SigLIP, non-discovery)
    for (const [pluginName, pluginResult] of Object.entries(result.results)) {
        if (pluginName === 'siglip') continue; // Already rendered
        if (discoveryResults[pluginName]) continue; // Already rendered from metadata
        renderPluginResultDiv(pluginName, pluginResult, '');
    }

    elements.lightboxTagBtn.textContent = 'Re-tag Image';
    elements.lightboxTagBtn.disabled = false;
}

function renderPluginResultDiv(pluginName, pluginResult, role) {
    if (!pluginResult) return;

    const pluginDiv = document.createElement('div');
    pluginDiv.className = 'plugin-result';

    const displayName = getPluginDisplayName(pluginName);
    const roleClass = role === 'scorer' ? 'role-scorer' : role === 'discovery' ? 'role-discovery' : '';

    if (pluginResult.error) {
        const header = document.createElement('div');
        header.className = 'plugin-header';
        const nameSpan = document.createElement('span');
        nameSpan.className = 'plugin-name';
        nameSpan.textContent = displayName;
        header.appendChild(nameSpan);

        const errorP = document.createElement('p');
        errorP.className = 'plugin-error';
        errorP.textContent = pluginResult.error;

        pluginDiv.appendChild(header);
        pluginDiv.appendChild(errorP);
    } else {
        const tags = pluginResult.tags || [];
        const timeMs = pluginResult.inference_time_ms || 0;

        // Build header
        const header = document.createElement('div');
        header.className = 'plugin-header';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'plugin-name';
        nameSpan.textContent = `${displayName}: ${tags.length} tags`;

        if (role) {
            const roleSpan = document.createElement('span');
            roleSpan.className = `plugin-role ${roleClass}`;
            roleSpan.textContent = role;
            header.appendChild(nameSpan);
            header.appendChild(roleSpan);
        } else {
            header.appendChild(nameSpan);
        }

        const timeSpan = document.createElement('span');
        timeSpan.className = 'plugin-time';
        timeSpan.textContent = `${timeMs.toFixed(0)}ms`;
        header.appendChild(timeSpan);

        // Build tags list
        const tagsList = document.createElement('div');
        tagsList.className = 'tags-list';

        // Render each tag
        for (const tag of tags) {
            const tagSpan = document.createElement('span');
            const hasConfidence = tag.confidence !== undefined && tag.confidence !== null;

            if (hasConfidence) {
                const conf = tag.confidence;
                const confClass = conf >= 0.8 ? 'high-confidence' : conf >= 0.6 ? 'medium-confidence' : '';
                tagSpan.className = `tag ${confClass}`;
                tagSpan.textContent = tag.label + ' ';
                const confSpan = document.createElement('span');
                confSpan.className = 'tag-confidence';
                confSpan.textContent = Math.round(conf * 100) + '%';
                tagSpan.appendChild(confSpan);
            } else {
                tagSpan.className = 'tag';
                tagSpan.textContent = tag.label;
            }
            tagsList.appendChild(tagSpan);
        }

        pluginDiv.appendChild(header);
        pluginDiv.appendChild(tagsList);
    }

    elements.lightboxTags.appendChild(pluginDiv);
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
    // Standard formats + RAW camera formats
    const imageExtensions = [
        // Standard formats
        '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif', '.heic', '.heif',
        // RAW camera formats
        '.arw',  // Sony
        '.cr2', '.cr3',  // Canon
        '.nef',  // Nikon
        '.dng',  // Adobe Digital Negative
        '.orf',  // Olympus
        '.rw2',  // Panasonic
        '.raf',  // Fujifilm
        '.pef',  // Pentax
        '.srw',  // Samsung
    ];
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

    // Update panel button if open for this image
    if (state.selectedImageId === imageId && state.panelOpen) {
        elements.panelTagBtn.disabled = true;
        elements.panelTagBtn.textContent = 'Processing...';
    }

    // Update lightbox button if open for this image
    if (state.selectedImageId === imageId && !elements.lightbox.hidden) {
        elements.lightboxTagBtn.disabled = true;
        elements.lightboxTagBtn.textContent = 'Processing...';
    }

    renderImageGrid();

    try {
        const result = await tagImage(imageId);

        image.results = result;
        image.processing = false;
        state.images.set(imageId, image);

        if (state.selectedImageId === imageId) {
            // Update panel if open
            if (state.panelOpen) {
                renderPanelTags(image);
            }
            // Update lightbox if open
            if (!elements.lightbox.hidden) {
                renderLightboxTags(image);
            }
        }

        renderImageGrid();
    } catch (error) {
        image.processing = false;
        state.images.set(imageId, image);
        showToast(`Tagging failed: ${error.message}`, 'error');

        if (state.selectedImageId === imageId) {
            // Reset panel button
            if (state.panelOpen) {
                elements.panelTagBtn.disabled = false;
                elements.panelTagBtn.textContent = 'Tag Image';
            }
            // Reset lightbox button
            if (!elements.lightbox.hidden) {
                elements.lightboxTagBtn.disabled = false;
                elements.lightboxTagBtn.textContent = 'Tag Image';
            }
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

/**
 * Get the current global quality level.
 * Returns "custom" if plugins have different quality settings.
 */
function getGlobalQuality() {
    const enabledPlugins = state.settings.plugins.filter(p => {
        const settings = state.settings.pluginSettings[p.name];
        return settings && p.available;
    });

    if (enabledPlugins.length === 0) {
        return 'standard'; // Default when no plugins
    }

    // Get quality of first plugin
    const firstSettings = state.settings.pluginSettings[enabledPlugins[0].name];
    const firstQuality = firstSettings?.quality || 'standard';

    // Check if all plugins have the same quality
    for (const plugin of enabledPlugins) {
        const settings = state.settings.pluginSettings[plugin.name];
        const quality = settings?.quality || 'standard';
        if (quality !== firstQuality) {
            return 'custom';
        }
    }

    return firstQuality;
}

/**
 * Update the global quality UI to reflect current state.
 */
function updateGlobalQualityUI() {
    const currentQuality = getGlobalQuality();

    // Update button active states
    elements.qualityButtons.forEach(btn => {
        const quality = btn.dataset.quality;
        btn.classList.toggle('active', quality === currentQuality);
    });

    // Update hint text
    if (currentQuality === 'custom') {
        elements.qualityHint.textContent = 'Plugins have different quality settings';
    } else {
        elements.qualityHint.textContent = 'Applies to all plugins';
    }
}

/**
 * Set quality for all available plugins.
 */
function setGlobalQuality(quality) {
    if (quality === 'custom') return; // Can't set "custom" directly

    for (const plugin of state.settings.plugins) {
        if (state.settings.pluginSettings[plugin.name] && plugin.available) {
            state.settings.pluginSettings[plugin.name].quality = quality;
        }
    }

    // Update the per-plugin dropdowns in the UI
    renderPluginsList();
    updateGlobalQualityUI();
    saveSettings();
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

// Detail Panel
elements.panelBack.addEventListener('click', closePanel);
elements.panelClose.addEventListener('click', closePanel);
elements.panelPreview.addEventListener('click', () => {
    if (state.selectedImageId) {
        openLightbox(state.selectedImageId);
    }
});
elements.panelViewBtn.addEventListener('click', () => {
    if (state.selectedImageId) {
        openLightbox(state.selectedImageId);
    }
});
elements.panelTagBtn.addEventListener('click', () => {
    if (state.selectedImageId) {
        tagSingleImage(state.selectedImageId);
    }
});

// Lightbox
elements.lightboxBack.addEventListener('click', closeLightbox);
elements.lightboxClose.addEventListener('click', closeLightbox);
elements.lightbox.querySelector('.lightbox-overlay').addEventListener('click', closeLightbox);
elements.lightboxTagBtn.addEventListener('click', () => {
    if (state.selectedImageId) {
        tagSingleImage(state.selectedImageId);
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        // Close modals/panels in order of priority
        if (!elements.lightbox.hidden) {
            closeLightbox();
        } else if (state.panelOpen) {
            closePanel();
        } else if (!elements.settingsModal.hidden) {
            closeSettings();
        }
    }
});

// Settings
elements.settingsBtn.addEventListener('click', openSettings);
elements.settingsClose.addEventListener('click', closeSettings);
elements.settingsModal.querySelector('.modal-overlay').addEventListener('click', closeSettings);

// Global quality buttons
elements.qualityButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const quality = btn.dataset.quality;
        if (quality !== 'custom') {
            setGlobalQuality(quality);
        }
    });
});


// ============================================================================
// Initialize
// ============================================================================

async function init() {
    // Fetch system status (initializes per-plugin settings)
    await fetchStatus();

    // Load any existing images from server
    await fetchExistingImages();

    // Initial render
    renderImageGrid();
}

// Start app
init();
