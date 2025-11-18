html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        padding: 20px;
                        background-color: #0e1117;
                        color: #fafafa;
                    }}
                    .audio-container {{
                        margin-bottom: 30px;
                        background-color: #1e1e1e;
                        padding: 20px;
                        border-radius: 10px;
                    }}
                    audio {{
                        width: 100%;
                        margin-bottom: 10px;
                    }}
                    .controls {{
                        display: flex;
                        gap: 10px;
                        margin-top: 10px;
                    }}
                    button {{
                        padding: 10px 20px;
                        font-size: 16px;
                        background-color: #ff4b4b;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                    }}
                    button:hover {{
                        background-color: #ff3333;
                    }}
                    .transcript-container {{
                        background-color: #1e1e1e;
                        padding: 30px;
                        border-radius: 10px;
                        min-height: 200px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-top: 20px;
                    }}
                    .transcript-display {{
                        text-align: center;
                        width: 100%;
                    }}
                    .speaker {{
                        font-size: 18px;
                        font-weight: bold;
                        color: #ff4b4b;
                        margin-bottom: 10px;
                    }}
                    .text {{
                        font-size: 24px;
                        line-height: 1.6;
                        color: #fafafa;
                        min-height: 60px;
                        padding: 10px;
                        border: 2px solid transparent;
                        border-radius: 5px;
                        cursor: text;
                        outline: none;
                    }}
                    .text:focus {{
                        border-color: #ff4b4b;
                        background-color: #2a2a2a;
                    }}
                    .text.editing {{
                        border-color: #ff4b4b;
                        background-color: #2a2a2a;
                    }}
                    .transcript-lines {{
                        display: flex;
                        flex-direction: column;
                        gap: 8px;
                        width: 100%;
                    }}
                    .transcript-line {{
                        font-size: 20px;
                        line-height: 1.6;
                        color: #aaa;
                        padding: 8px;
                        border-radius: 5px;
                        transition: all 0.3s ease;
                    }}
                    .transcript-line.current {{
                        font-weight: bold;
                        font-size: 24px;
                        color: #fafafa;
                        background-color: rgba(255, 75, 75, 0.1);
                        border-left: 3px solid #ff4b4b;
                        padding-left: 15px;
                    }}
                    .transcript-line.editable {{
                        cursor: text;
                    }}
                    .transcript-line.editable:hover {{
                        background-color: rgba(255, 255, 255, 0.05);
                    }}
                    .transcript-line.editing {{
                        border: 2px solid #ff4b4b;
                        background-color: #2a2a2a;
                        color: #fafafa;
                    }}
                    .time-display {{
                        font-size: 14px;
                        color: #888;
                        margin-top: 10px;
                    }}
                    .fade-in {{
                        animation: fadeIn 0.5s;
                    }}
                    .save-button {{
                        background-color: #4CAF50 !important;
                        margin-top: 0;
                        padding: 12px 24px !important;
                        font-size: 16px !important;
                        font-weight: bold;
                        color: white !important;
                        border: none !important;
                        border-radius: 5px;
                        cursor: pointer;
                        width: auto;
                        min-width: 150px;
                        flex-shrink: 0;
                        display: inline-block;
                        box-sizing: border-box;
                    }}
                    .save-button:hover {{
                        background-color: #45a049;
                    }}
                    .resume-button {{
                        background-color: #2196F3;
                    }}
                    .resume-button:hover {{
                        background-color: #0b7dda;
                    }}
                    .edit-notice {{
                        font-size: 12px;
                        color: #ffa500;
                        margin-top: 5px;
                        font-style: italic;
                    }}
                    .transcript-line-content {{
                        display: flex;
                        align-items: center;
                        flex-wrap: wrap;
                        gap: 8px;
                    }}
                    .transcript-text-part {{
                        flex: 1;
                        min-width: 200px;
                    }}
                    .transcript-text-part.editable {{
                        cursor: text;
                    }}
                    .transcript-text-part.editing {{
                        border: 2px solid #ff4b4b;
                        background-color: #2a2a2a;
                        color: #fafafa;
                        padding: 4px;
                        border-radius: 3px;
                    }}
                    .emotions-container {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 6px;
                        align-items: center;
                    }}
                    .emotion-box {{
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        background-color: rgba(255, 75, 75, 0.2);
                        color: #fafafa;
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-size: 14px;
                        border: 1px solid rgba(255, 75, 75, 0.4);
                        min-height: 32px;
                    }}
                    .emotion-remove {{
                        cursor: pointer;
                        font-weight: bold;
                        color: #ff8888;
                        padding: 4px 6px;
                        border: none;
                        background: none;
                        font-size: 20px;
                        line-height: 1;
                        min-width: 28px;
                        min-height: 28px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 4px;
                        transition: all 0.2s ease;
                    }}
                    .emotion-remove:hover {{
                        color: #ff4b4b;
                        background-color: rgba(255, 75, 75, 0.2);
                    }}
                    .add-emotion-btn {{
                        cursor: pointer;
                        background-color: rgba(75, 175, 80, 0.2);
                        color: #4CAF50;
                        border: 1px solid rgba(75, 175, 80, 0.4);
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-size: 18px;
                        font-weight: bold;
                        min-width: 32px;
                        min-height: 32px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: all 0.2s ease;
                    }}
                    .add-emotion-btn:hover {{
                        background-color: rgba(75, 175, 80, 0.3);
                        transform: scale(1.05);
                    }}
                    .emotion-modal {{
                        display: none;
                        position: fixed;
                        z-index: 99999;
                        left: 0;
                        top: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0, 0, 0, 0.7);
                        overflow: auto;
                    }}
                    .emotion-modal-content {{
                        background-color: #1e1e1e;
                        margin: 5% auto;
                        padding: 20px;
                        border: 1px solid #444;
                        border-radius: 10px;
                        width: 80%;
                        max-width: 600px;
                        max-height: 80vh;
                        overflow-y: auto;
                    }}
                    .emotion-modal-header {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 20px;
                    }}
                    .emotion-modal-title {{
                        font-size: 18px;
                        font-weight: bold;
                        color: #fafafa;
                    }}
                    .emotion-modal-close {{
                        cursor: pointer;
                        font-size: 24px;
                        color: #aaa;
                        border: none;
                        background: none;
                    }}
                    .emotion-modal-close:hover {{
                        color: #fafafa;
                    }}
                    .emotion-options {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 10px;
                        margin-top: 10px;
                    }}
                    .emotion-option {{
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        padding: 8px;
                        background-color: #2a2a2a;
                        border-radius: 5px;
                        cursor: pointer;
                    }}
                    .emotion-option:hover {{
                        background-color: #333;
                    }}
                    .emotion-option input[type="checkbox"] {{
                        width: 18px;
                        height: 18px;
                        cursor: pointer;
                    }}
                    .emotion-option label {{
                        cursor: pointer;
                        color: #fafafa;
                        flex: 1;
                    }}
                    .emotion-modal-actions {{
                        margin-top: 20px;
                        display: flex;
                        justify-content: flex-end;
                        gap: 10px;
                    }}
                    .emotion-modal-btn {{
                        padding: 10px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 14px;
                    }}
                    .emotion-modal-btn-primary {{
                        background-color: #4CAF50;
                        color: white;
                    }}
                    .emotion-modal-btn-primary:hover {{
                        background-color: #45a049;
                    }}
                    .emotion-modal-btn-secondary {{
                        background-color: #666;
                        color: white;
                    }}
                    .emotion-modal-btn-secondary:hover {{
                        background-color: #777;
                    }}
                    .intensity-container {{
                        display: flex;
                        align-items: center;
                        gap: 4px;
                        margin-left: 8px;
                    }}
                    .intensity-label {{
                        font-size: 12px;
                        color: #aaa;
                        margin-right: 4px;
                    }}
                    .intensity-button {{
                        width: 28px;
                        height: 28px;
                        border-radius: 50%;
                        border: 2px solid #666;
                        background-color: #2a2a2a;
                        color: #fafafa;
                        font-size: 14px;
                        font-weight: bold;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: all 0.2s ease;
                        padding: 0;
                        min-width: 28px;
                    }}
                    .intensity-button:hover {{
                        border-color: #ff4b4b;
                        background-color: #3a3a3a;
                        transform: scale(1.1);
                    }}
                    .intensity-button.selected {{
                        background-color: #ff4b4b;
                        border-color: #ff4b4b;
                        color: white;
                    }}
                    @keyframes fadeIn {{
                        from {{ opacity: 0; transform: translateY(10px); }}
                        to {{ opacity: 1; transform: translateY(0); }}
                    }}
                </style>
            </head>
            <body>
                <div class="audio-container">
                    <audio id="audioPlayer" controls>
                        <source src="data:audio/{audio_format};base64,{audio_base64}" type="audio/{audio_format}">
                        Your browser does not support the audio element.
                    </audio>
                    <div class="controls">
                        <button onclick="playAudio()">▶ Play</button>
                        <button onclick="pauseAudio()">⏸ Pause</button>
                        <button onclick="goBack5Seconds()">⏪ Go Back 5s</button>
                        <button id="speedButton" onclick="togglePlaybackSpeed()">1.0x Speed</button>
                        <button id="resumeButton" onclick="resumeAudio()" class="resume-button" style="display: none;">▶ Resume</button>
                    </div>
                </div>
                
                <div class="transcript-container">
                    <div class="transcript-display" id="transcriptDisplay">
                        <div class="speaker" id="speakerDisplay">Ready to play...</div>
                        <div class="transcript-lines" id="transcriptLines">
                            <div class="transcript-line">Click Play to start</div>
                        </div>
                        <div class="edit-notice" id="editNotice" style="display: none;">Editing mode - Audio paused. Click Resume to continue.</div>
                        <div class="time-display" id="timeDisplay">00:00:00</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 20px; margin-bottom: 20px; position: sticky; bottom: 0; background-color: #0e1117; padding: 10px 0; z-index: 100;">
                    <button onclick="prepareSave()" class="save-button">💾 Prepare Save</button>
                </div>
                
                <!-- Emotion Modal -->
                <div id="emotionModal" class="emotion-modal">
                    <div class="emotion-modal-content">
                        <div class="emotion-modal-header">
                            <div class="emotion-modal-title">Select Emotions</div>
                            <button class="emotion-modal-close" onclick="closeEmotionModal()">&times;</button>
                        </div>
                        <div class="emotion-options" id="emotionOptions">
                            <!-- Will be populated by JavaScript -->
                        </div>
                        <div class="emotion-modal-actions">
                            <button class="emotion-modal-btn emotion-modal-btn-secondary" onclick="closeEmotionModal()">Cancel</button>
                            <button class="emotion-modal-btn emotion-modal-btn-primary" onclick="applySelectedEmotions()">Apply</button>
                        </div>
                    </div>
                </div>
                
                <!-- Transcript Copy Modal -->
                <div id="transcriptCopyModal" class="emotion-modal">
                    <div class="emotion-modal-content" style="max-width: 800px;">
                        <div class="emotion-modal-header">
                            <div class="emotion-modal-title">📋 Transcript Ready - Copy to Streamlit Text Area</div>
                            <button class="emotion-modal-close" onclick="closeTranscriptCopyModal()">&times;</button>
                        </div>
                        <div style="margin: 20px 0;">
                            <p style="color: #fafafa; margin-bottom: 15px;">✅ Transcript copied to clipboard! The text is also shown below. Click in the Streamlit text area below and press <strong>Ctrl+V</strong> (or <strong>Cmd+V</strong> on Mac) to paste.</p>
                            <textarea id="transcriptTextArea" readonly style="width: 100%; min-height: 300px; padding: 15px; background-color: #2a2a2a; color: #fafafa; border: 1px solid #444; border-radius: 5px; font-family: monospace; font-size: 12px; resize: vertical; overflow-y: auto;" placeholder="Transcript will appear here..."></textarea>
                        </div>
                        <div class="emotion-modal-actions">
                            <button class="emotion-modal-btn emotion-modal-btn-secondary" onclick="copyTranscriptAgain()">📋 Copy Again</button>
                            <button class="emotion-modal-btn emotion-modal-btn-primary" onclick="closeTranscriptCopyModal()">Got It</button>
                        </div>
                    </div>
                </div>
                
                <script>
                    const transcriptData = {transcript_json};
                    const audio = document.getElementById('audioPlayer');
                    let currentEntryIndex = 0;
                    let updateInterval = null;
                    let isEditing = false;
                    let editedEntries = {{}}; // Store edited entries by index
                    let editedEmotions = {{}}; // Store edited emotions by index
                    let editedIntensities = {{}}; // Store edited intensities by index
                    let wasPlayingBeforeEdit = false;
                    let playbackSpeed = 0.75; // Track current playback speed (default 0.75x)
                    let currentEmotionEntryIndex = null; // Track which entry is having emotions edited
                    
                    // Available emotions
                    const availableEmotions = [
                        'Adoration', 'Admiration', 'Amusement', 'Anger',
                        'Anxiety', 'Awe', 'Awkwardness', 'Boredom',
                        'Calmness', 'Confusion', 'Concentration', 'Contempt',
                        'Contentment', 'Contemplation', 'Craving', 'Desire',
                        'Determination', 'Disappointment', 'Disapproval', 'Disgust',
                        'Distress', 'Doubt', 'Ecstasy', 'Embarrassment',
                        'Excitement', 'Fear', 'Interest', 'Joy',
                        'Love', 'Nostalgia', 'Pride', 'Realization',
                        'Relief', 'Sadness', 'Satisfaction', 'Surprise (Negative)',
                        'Surprise (Positive)', 'Tiredness', 'Trust', 'Wonder'
                    ];
       
                    // Set default playback speed to 0.75x
                    audio.playbackRate = 0.75;
                    document.getElementById('speedButton').textContent = '0.75x Speed';
                    
                    // Initialize emotion modal options
                    function initializeEmotionModal() {{
                        const optionsContainer = document.getElementById('emotionOptions');
                        optionsContainer.innerHTML = '';
                        availableEmotions.forEach(emotion => {{
                            const optionDiv = document.createElement('div');
                            optionDiv.className = 'emotion-option';
                            const checkbox = document.createElement('input');
                            checkbox.type = 'checkbox';
                            checkbox.id = 'emotion-' + emotion.replace(/\\s+/g, '-');
                            checkbox.value = emotion;
                            const label = document.createElement('label');
                            label.htmlFor = checkbox.id;
                            label.textContent = emotion;
                            optionDiv.appendChild(checkbox);
                            optionDiv.appendChild(label);
                            optionsContainer.appendChild(optionDiv);
                        }});
                    }}
                    
                    // Initialize on load
                    initializeEmotionModal();
                    
                    // Wire up modal buttons after DOM is ready
                    document.addEventListener('DOMContentLoaded', function() {{
                        const closeBtn = document.querySelector('.emotion-modal-close');
                        if (closeBtn) {{
                            closeBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                closeEmotionModal();
                            }});
                        }}
                        
                        const cancelBtn = document.querySelector('.emotion-modal-btn-secondary');
                        if (cancelBtn) {{
                            cancelBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                closeEmotionModal();
                            }});
                        }}
                        
                        const applyBtn = document.querySelector('.emotion-modal-btn-primary');
                        if (applyBtn) {{
                            applyBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                applySelectedEmotions();
                            }});
                        }}
                    }});
                    
                    // Also try to wire up immediately (in case DOM is already loaded)
                    setTimeout(function() {{
                        const closeBtn = document.querySelector('.emotion-modal-close');
                        if (closeBtn && !closeBtn.dataset.wired) {{
                            closeBtn.dataset.wired = 'true';
                            closeBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                closeEmotionModal();
                            }});
                        }}
                        
                        const cancelBtn = document.querySelector('.emotion-modal-btn-secondary');
                        if (cancelBtn && !cancelBtn.dataset.wired) {{
                            cancelBtn.dataset.wired = 'true';
                            cancelBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                closeEmotionModal();
                            }});
                        }}
                        
                        const applyBtn = document.querySelector('.emotion-modal-btn-primary');
                        if (applyBtn && !applyBtn.dataset.wired) {{
                            applyBtn.dataset.wired = 'true';
                            applyBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                applySelectedEmotions();
                            }});
                        }}
                    }}, 100);
                    
                    function formatTime(seconds) {{
                        const h = Math.floor(seconds / 3600);
                        const m = Math.floor((seconds % 3600) / 60);
                        const s = Math.floor(seconds % 60);
                        const ms = Math.floor((seconds % 1) * 100);
                        return `${{h.toString().padStart(2, '0')}}:${{m.toString().padStart(2, '0')}}:${{s.toString().padStart(2, '0')}}.${{ms.toString().padStart(2, '0')}}`;
                    }}
                    
                    function formatTimeForLine(seconds) {{
                        const h = Math.floor(seconds / 3600);
                        const m = Math.floor((seconds % 3600) / 60);
                        const s = seconds % 60;
                        return `${{h.toString().padStart(2, '0')}}:${{m.toString().padStart(2, '0')}}:${{s.toFixed(3).padStart(6, '0')}}`;
                    }}
                    
                    function updateTranscript(forceUpdate = false) {{
                        if (isEditing && !forceUpdate) {{
                            // Don't update while editing - but keep the current display
                            // Unless forceUpdate is true (for emotion changes)
                            return;
                        }}
                        
                        const currentTime = audio.currentTime;
                        const timeDisplay = document.getElementById('timeDisplay');
                        timeDisplay.textContent = formatTime(currentTime);
                        
                        // Find the current entry based on time
                        let activeEntry = null;
                        let newEntryIndex = currentEntryIndex; // Preserve current index if force updating
                        
                        if (!forceUpdate) {{
                            // Only update entry index if not forcing (normal time-based update)
                            for (let i = transcriptData.length - 1; i >= 0; i--) {{
                                if (currentTime >= transcriptData[i].time) {{
                                    activeEntry = transcriptData[i];
                                    newEntryIndex = i;
                                    break;
                                }}
                            }}
                            
                            // If we're past the last entry, show the last one
                            if (!activeEntry && transcriptData.length > 0) {{
                                activeEntry = transcriptData[transcriptData.length - 1];
                                newEntryIndex = transcriptData.length - 1;
                            }}
                            
                            // If we're before the first entry, show nothing or first entry
                            if (!activeEntry && transcriptData.length > 0 && currentTime < transcriptData[0].time) {{
                                document.getElementById('speakerDisplay').textContent = 'Waiting...';
                                const transcriptLinesContainer = document.getElementById('transcriptLines');
                                transcriptLinesContainer.innerHTML = '<div class="transcript-line">Audio will start soon</div>';
                                return;
                            }}
                            
                            currentEntryIndex = newEntryIndex;
                        }} else {{
                            // When forcing update, use current entry index
                            if (currentEntryIndex >= 0 && currentEntryIndex < transcriptData.length) {{
                                activeEntry = transcriptData[currentEntryIndex];
                            }}
                        }}
                        
                        // If we still don't have an active entry, use first one
                        if (!activeEntry && transcriptData.length > 0) {{
                            activeEntry = transcriptData[0];
                            currentEntryIndex = 0;
                        }}
                        
                        if (activeEntry) {{
                            const speakerDisplay = document.getElementById('speakerDisplay');
                            const transcriptLinesContainer = document.getElementById('transcriptLines');
                            
                            // Show speaker name
                            speakerDisplay.textContent = activeEntry.speaker;
                            
                            // Get previous 2, current, and next 2 entries
                            const startIndex = Math.max(0, currentEntryIndex - 2);
                            const endIndex = Math.min(transcriptData.length - 1, currentEntryIndex + 2);
                            
                            // Clear previous lines
                            transcriptLinesContainer.innerHTML = '';
                            
                            // Create lines for visible entries
                            for (let i = startIndex; i <= endIndex; i++) {{
                                const entry = transcriptData[i];
                                const isCurrent = (i === currentEntryIndex);
                                
                                // Use edited text if available, otherwise use original
                                const displayText = editedEntries[i] || entry.text;
                                
                                // Get emotions for this entry (use edited if available, otherwise original)
                                let entryEmotions = editedEmotions[i];
                                if (entryEmotions === undefined) {{
                                    entryEmotions = entry.emotions || [];
                                }}
                                
                                // Create line element
                                const lineDiv = document.createElement('div');
                                lineDiv.className = 'transcript-line' + (isCurrent ? ' current' : '');
                                lineDiv.setAttribute('data-index', i);
                                
                                // Create content container
                                const contentDiv = document.createElement('div');
                                contentDiv.className = 'transcript-line-content';
                                
                                // Create text part
                                const textPart = document.createElement('div');
                                textPart.className = 'transcript-text-part';
                                
                                if (isCurrent) {{
                                    // Make current line editable
                                    textPart.contentEditable = 'true';
                                    textPart.classList.add('editable');
                                    
                                    // Handle focus event to start editing mode
                                    textPart.addEventListener('focus', function() {{
                                        if (!isEditing) {{
                                            onTextClick(i);
                                        }}
                                    }}, true);
                                    
                                    // Handle blur event to save edits
                                    textPart.addEventListener('blur', function() {{
                                        onTextBlur(i);
                                    }});
                                    
                                    // Handle click to ensure editing starts
                                    textPart.addEventListener('click', function(e) {{
                                        if (!isEditing) {{
                                            onTextClick(i);
                                        }}
                                    }});
                                }}
                                
                                textPart.textContent = entry.speaker + ': ' + displayText;
                                
                                // Create emotions container
                                const emotionsContainer = document.createElement('div');
                                emotionsContainer.className = 'emotions-container';
                                // Prevent clicks on emotion container from triggering text editing
                                emotionsContainer.addEventListener('click', function(e) {{
                                    e.stopPropagation();
                                }});
                                
                                // Add emotion boxes
                                entryEmotions.forEach(emotion => {{
                                    const emotionBox = document.createElement('div');
                                    emotionBox.className = 'emotion-box';
                                    
                                    const emotionText = document.createElement('span');
                                    emotionText.textContent = emotion;
                                    
                                    const removeBtn = document.createElement('button');
                                    removeBtn.className = 'emotion-remove';
                                    removeBtn.textContent = '×';
                                    removeBtn.type = 'button'; // Prevent form submission
                                    removeBtn.addEventListener('click', function(e) {{
                                        e.preventDefault();
                                        e.stopPropagation();
                                        removeEmotion(i, emotion);
                                    }});
                                    
                                    emotionBox.appendChild(emotionText);
                                    emotionBox.appendChild(removeBtn);
                                    emotionsContainer.appendChild(emotionBox);
                                }});
                                
                                // Add + button to add emotions
                                const addEmotionBtn = document.createElement('button');
                                addEmotionBtn.className = 'add-emotion-btn';
                                addEmotionBtn.textContent = '+';
                                addEmotionBtn.type = 'button'; // Prevent form submission
                                addEmotionBtn.addEventListener('click', function(e) {{
                                    e.preventDefault();
                                    e.stopPropagation();
                                    openEmotionModal(i);
                                }});
                                emotionsContainer.appendChild(addEmotionBtn);
                                
                                // Create intensity container
                                const intensityContainer = document.createElement('div');
                                intensityContainer.className = 'intensity-container';
                                // Prevent clicks on intensity container from triggering text editing
                                intensityContainer.addEventListener('click', function(e) {{
                                    e.stopPropagation();
                                }});
                                
                                // Get intensity for this entry (use edited if available, otherwise original)
                                let entryIntensity = editedIntensities[i];
                                if (entryIntensity === undefined) {{
                                    entryIntensity = entry.intensity !== undefined ? entry.intensity : 3;
                                }}
                                
                                // Create intensity buttons (1-5)
                                for (let intensity = 1; intensity <= 5; intensity++) {{
                                    const intensityBtn = document.createElement('button');
                                    intensityBtn.className = 'intensity-button' + (entryIntensity === intensity ? ' selected' : '');
                                    intensityBtn.textContent = intensity;
                                    intensityBtn.type = 'button';
                                    intensityBtn.setAttribute('data-intensity', intensity);
                                    intensityBtn.addEventListener('click', function(e) {{
                                        e.preventDefault();
                                        e.stopPropagation();
                                        setIntensity(i, intensity);
                                    }});
                                    intensityContainer.appendChild(intensityBtn);
                                }}
                                
                                // Assemble the line
                                contentDiv.appendChild(textPart);
                                contentDiv.appendChild(emotionsContainer);
                                contentDiv.appendChild(intensityContainer);
                                lineDiv.appendChild(contentDiv);
                                
                                transcriptLinesContainer.appendChild(lineDiv);
                            }}
                        }}
                    }}
                    
                    function onTextClick(entryIndex) {{
                        // Only allow editing the current line
                        if (entryIndex !== currentEntryIndex) {{
                            return;
                        }}
                        
                        if (!isEditing) {{
                            isEditing = true;
                            wasPlayingBeforeEdit = !audio.paused;
                            
                            // Pause audio
                            if (wasPlayingBeforeEdit) {{
                                audio.pause();
                            }}
                            
                            const editNotice = document.getElementById('editNotice');
                            const resumeButton = document.getElementById('resumeButton');
                            
                            // Find and highlight the editable text part
                            const textParts = document.querySelectorAll('.transcript-text-part.editable');
                            textParts.forEach(textPart => {{
                                const lineDiv = textPart.closest('.transcript-line');
                                if (lineDiv) {{
                                    const lineIndex = parseInt(lineDiv.getAttribute('data-index'));
                                    if (lineIndex === entryIndex) {{
                                        textPart.classList.add('editing');
                                    }}
                                }}
                            }});
                            
                            editNotice.style.display = 'block';
                            resumeButton.style.display = 'inline-block';
                        }}
                    }}
                    
                    function onTextBlur(entryIndex) {{
                        // Save the edited text
                        const textParts = document.querySelectorAll('.transcript-text-part.editable');
                        textParts.forEach(textPart => {{
                            const lineDiv = textPart.closest('.transcript-line');
                            if (lineDiv) {{
                                const lineIndex = parseInt(lineDiv.getAttribute('data-index'));
                                if (lineIndex === entryIndex) {{
                                    const editedText = textPart.textContent.trim();
                                    // Remove speaker prefix if present
                                    const speakerPrefix = transcriptData[entryIndex].speaker + ': ';
                                    const cleanText = editedText.startsWith(speakerPrefix) 
                                        ? editedText.substring(speakerPrefix.length) 
                                        : editedText;
                                    
                                    if (cleanText && cleanText !== transcriptData[entryIndex].text) {{
                                        editedEntries[entryIndex] = cleanText;
                                    }}
                                    textPart.classList.remove('editing');
                                }}
                            }}
                        }});
                    }}
                    
                    function openEmotionModal(entryIndex) {{
                        console.log('Opening emotion modal for entry:', entryIndex);
                        currentEmotionEntryIndex = entryIndex;
                        
                        // Get current emotions for this entry
                        let currentEmotions = editedEmotions[entryIndex];
                        if (currentEmotions === undefined) {{
                            currentEmotions = transcriptData[entryIndex].emotions || [];
                        }}
                        
                        console.log('Current emotions:', currentEmotions);
                        
                        // Check the checkboxes for current emotions
                        const checkboxes = document.querySelectorAll('#emotionOptions input[type="checkbox"]');
                        checkboxes.forEach(checkbox => {{
                            checkbox.checked = currentEmotions.includes(checkbox.value);
                        }});
                        
                        // Show modal
                        const modal = document.getElementById('emotionModal');
                        if (modal) {{
                            modal.style.display = 'block';
                            console.log('Modal displayed');
                        }} else {{
                            console.error('Modal element not found!');
                        }}
                    }}
                    
                    function closeEmotionModal() {{
                        const modal = document.getElementById('emotionModal');
                        if (modal) {{
                            modal.style.display = 'none';
                        }}
                        currentEmotionEntryIndex = null;
                    }}
                    
                    function applySelectedEmotions() {{
                        if (currentEmotionEntryIndex === null) {{
                            console.log('No entry index set');
                            return;
                        }}
                        
                        // Get selected emotions
                        const checkboxes = document.querySelectorAll('#emotionOptions input[type="checkbox"]:checked');
                        const selectedEmotions = Array.from(checkboxes).map(cb => cb.value);
                        
                        console.log('Applying emotions:', selectedEmotions, 'to entry:', currentEmotionEntryIndex);
                        
                        // Save to editedEmotions
                        editedEmotions[currentEmotionEntryIndex] = selectedEmotions;
                        
                        // Force update display immediately
                        updateTranscript(true);
                        
                        // Close modal
                        closeEmotionModal();
                    }}
                    
                    function removeEmotion(entryIndex, emotionToRemove) {{
                        console.log('Removing emotion:', emotionToRemove, 'from entry:', entryIndex);
                        
                        // Get current emotions for this entry
                        let currentEmotions = editedEmotions[entryIndex];
                        if (currentEmotions === undefined) {{
                            currentEmotions = [...(transcriptData[entryIndex].emotions || [])];
                        }} else {{
                            currentEmotions = [...currentEmotions];
                        }}
                        
                        console.log('Current emotions before removal:', currentEmotions);
                        
                        // Remove the emotion
                        currentEmotions = currentEmotions.filter(e => e !== emotionToRemove);
                        
                        console.log('Current emotions after removal:', currentEmotions);
                        
                        // Save back
                        editedEmotions[entryIndex] = currentEmotions;
                        
                        // Force update display immediately
                        updateTranscript(true);
                    }}
                    
                    function setIntensity(entryIndex, intensity) {{
                        console.log('Setting intensity:', intensity, 'for entry:', entryIndex);
                        
                        // Save intensity
                        editedIntensities[entryIndex] = intensity;
                        
                        // Force update display immediately
                        updateTranscript(true);
                    }}
                    
                    
                    function resumeAudio() {{
                        isEditing = false;
                        
                        const editNotice = document.getElementById('editNotice');
                        const resumeButton = document.getElementById('resumeButton');
                        
                        // Remove editing class from all text parts
                        const editingTextParts = document.querySelectorAll('.transcript-text-part.editing');
                        editingTextParts.forEach(textPart => {{
                            const lineDiv = textPart.closest('.transcript-line');
                            if (lineDiv) {{
                                const entryIndex = parseInt(lineDiv.getAttribute('data-index'));
                                if (entryIndex !== null) {{
                                    onTextBlur(entryIndex);
                                }}
                            }}
                        }});
                        
                        editNotice.style.display = 'none';
                        resumeButton.style.display = 'none';
                        
                        // Resume playback if it was playing before
                        if (wasPlayingBeforeEdit) {{
                            audio.play();
                        }}
                    }}
                    
                    function playAudio() {{
                        if (isEditing) {{
                            resumeAudio();
                        }}
                        audio.play();
                        if (!updateInterval) {{
                            updateInterval = setInterval(updateTranscript, 100);
                        }}
                        updateTranscript();
                    }}
                    
                    function pauseAudio() {{
                        audio.pause();
                    }}
                    
                    function stopAudio() {{
                        audio.pause();
                        audio.currentTime = 0;
                        currentEntryIndex = 0;
                        isEditing = false;
                        document.getElementById('speakerDisplay').textContent = 'Stopped';
                        const transcriptLinesContainer = document.getElementById('transcriptLines');
                        transcriptLinesContainer.innerHTML = '<div class="transcript-line">Click Play to start</div>';
                        document.getElementById('editNotice').style.display = 'none';
                        document.getElementById('resumeButton').style.display = 'none';
                        document.getElementById('timeDisplay').textContent = '00:00:00';
                    }}
                    
                    function goBack5Seconds() {{
                        // Go back 5 seconds, but don't go below 0
                        const newTime = Math.max(0, audio.currentTime - 5);
                        audio.currentTime = newTime;
                        
                        // Immediately update transcript to show what was 5 seconds ago
                        updateTranscript();
                    }}
                    
                    function togglePlaybackSpeed() {{
                        // Toggle between 1.0x (default) and 0.75x speed
                        if (playbackSpeed === 0.75) {{
                            playbackSpeed = 1.0;
                            audio.playbackRate = 1.0;
                            document.getElementById('speedButton').textContent = '1.0x Speed';
                        }} else {{
                            playbackSpeed = 0.75;
                            audio.playbackRate = 0.75;
                            document.getElementById('speedButton').textContent = '0.75x Speed';
                        }}
                    }}
                    
                    function prepareSave() {{
                        // Build the edited transcript in the original format
                        let editedLines = [];
                        
                        for (let i = 0; i < transcriptData.length; i++) {{
                            const entry = transcriptData[i];
                            const editedText = editedEntries[i] || entry.text;
                            
                            // Get emotions (use edited if available, otherwise original)
                            let entryEmotions = editedEmotions[i];
                            if (entryEmotions === undefined) {{
                                entryEmotions = entry.emotions || [];
                            }}
                            
                            // Get intensity (use edited if available, otherwise original, default 3)
                            let entryIntensity = editedIntensities[i];
                            if (entryIntensity === undefined) {{
                                entryIntensity = entry.intensity !== undefined ? entry.intensity : 3;
                            }}
                            
                            // Reconstruct the line in original format
                            let timestamp;
                            if (entry.timestamp_format === 'range_format') {{
                                // Use [start,end] format
                                timestamp = `${{entry.time.toFixed(3)}},${{entry.end_time.toFixed(3)}}`;
                            }} else {{
                                // Use [HH:MM:SS.mmm] format
                                timestamp = formatTimeForLine(entry.time);
                            }}
                            
                            // Format intensity as [Intensity: X]
                            let intensityPart = ` [Intensity: ${{entryIntensity}}]`;
                            
                            // Format emotions as (Emotion: Emotion1, Emotion2) or empty string
                            let emotionPart = '';
                            if (entryEmotions.length > 0) {{
                                emotionPart = ' (Emotion: ' + entryEmotions.join(', ') + ')';
                            }}
                            
                            const line = `[${{timestamp}}] ${{entry.speaker}}: ${{editedText}}${{intensityPart}}${{emotionPart}}`;
                            
                            editedLines.push(line);
                        }}
                        
                        const transcriptText = editedLines.join('\\n');
                        
                        // Populate the modal textarea
                        const textarea = document.getElementById('transcriptTextArea');
                        if (textarea) {{
                            textarea.value = transcriptText;
                        }}
                        
                        // Copy to clipboard automatically
                        navigator.clipboard.writeText(transcriptText).then(function() {{
                            // Show the modal
                            const modal = document.getElementById('transcriptCopyModal');
                            if (modal) {{
                                modal.style.display = 'block';
                                // Auto-select text in textarea for easy copying
                                if (textarea) {{
                                    textarea.select();
                                    textarea.setSelectionRange(0, transcriptText.length);
                                }}
                            }}
                        }}, function(err) {{
                            // Fallback if clipboard API fails - still show modal
                            const modal = document.getElementById('transcriptCopyModal');
                            if (modal) {{
                                modal.style.display = 'block';
                                if (textarea) {{
                                    textarea.select();
                                    textarea.setSelectionRange(0, transcriptText.length);
                                }}
                            }}
                        }});
                    }}
                    
                    function closeTranscriptCopyModal() {{
                        const modal = document.getElementById('transcriptCopyModal');
                        if (modal) {{
                            modal.style.display = 'none';
                        }}
                    }}
                    
                    function copyTranscriptAgain() {{
                        const textarea = document.getElementById('transcriptTextArea');
                        if (textarea) {{
                            const text = textarea.value;
                            navigator.clipboard.writeText(text).then(function() {{
                                alert('✅ Transcript copied to clipboard again!');
                            }}, function(err) {{
                                // Select text as fallback
                                textarea.select();
                                textarea.setSelectionRange(0, text.length);
                                alert('Please copy the text manually (Ctrl+C or Cmd+C)');
                            }});
                        }}
                    }}
                    
                    // Close transcript modal when clicking outside
                    window.onclick = function(event) {{
                        const emotionModal = document.getElementById('emotionModal');
                        const transcriptModal = document.getElementById('transcriptCopyModal');
                        if (event.target === emotionModal) {{
                            closeEmotionModal();
                        }}
                        if (event.target === transcriptModal) {{
                            closeTranscriptCopyModal();
                        }}
                    }}
                    
                    // Update transcript when audio time changes
                    audio.addEventListener('timeupdate', updateTranscript);
                    
                    // Initialize
                    updateTranscript();
                </script>
            </body>
            </html>
            """