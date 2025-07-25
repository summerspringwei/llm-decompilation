require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.36.1/min/vs' }});

require(['vs/editor/editor.main'], function() {
    // Define LLVM IR language
    monaco.languages.register({ id: 'llvm' });

    // Define Assembly language
    monaco.languages.register({ id: 'asm' });

    // Define LLVM IR tokens
    monaco.languages.setMonarchTokensProvider('llvm', {
        tokenizer: {
            root: [
                // Keywords
                [/\b(define|declare|global|constant|private|internal|linkonce|linkonce_odr|weak|weak_odr|appending|dllimport|dllexport|common|available_externally|default|hidden|protected|extern_weak|external|thread_local|zeroinitializer|undef|null|to|tail|target|triple|datalayout|volatile|nuw|nsw|nnan|ninf|nsz|arcp|fast|exact|inbounds|align|addrspace|section|alias|module|asm|sideeffect|gc|dbg)\b/, 'keyword'],
                
                // Types
                [/\b(void|half|float|double|x86_fp80|fp128|ppc_fp128|label|metadata|token|x86_mmx|i8|i16|i32|i64|i128)\b/, 'type'],
                
                // Instructions
                [/\b(add|alloca|and|ashr|br|call|fadd|fcmp|fdiv|fmul|fpext|fptosi|fptoui|fptrunc|frem|fsub|getelementptr|icmp|indirectbr|invoke|landingpad|load|lshr|mul|or|phi|ptrtoint|ret|sdiv|select|sext|shl|sitofp|srem|store|sub|switch|trunc|udiv|uitofp|urem|xor|zext)\b/, 'keyword.control'],
                
                // Variables and labels
                [/[%@][-a-zA-Z$._][-a-zA-Z$._0-9]*/, 'variable'],
                
                // Numbers
                [/\b[0-9]+\b/, 'number'],
                
                // Strings
                [/"[^"]*"/, 'string'],
                
                // Comments
                [/;.*$/, 'comment'],
                
                // Operators
                [/[+\-*/=<>!&|^~]+/, 'operator'],
                
                // Brackets and punctuation
                [/[{}()[\]]/, 'delimiter.bracket'],
                [/[,:.]/, 'delimiter']
            ]
        }
    });

    // Define Assembly tokens
    monaco.languages.setMonarchTokensProvider('asm', {
        tokenizer: {
            root: [
                // Instructions
                [/\b(mov|add|sub|mul|div|push|pop|call|ret|jmp|je|jne|jl|jle|jg|jge|cmp|test|and|or|xor|not|shl|shr|lea|inc|dec)\b/i, 'keyword'],
                
                // Registers
                [/\b(eax|ebx|ecx|edx|esi|edi|esp|ebp|rax|rbx|rcx|rdx|rsi|rdi|rsp|rbp|r8|r9|r10|r11|r12|r13|r14|r15)\b/i, 'type'],
                
                // Labels
                [/[a-zA-Z_][a-zA-Z0-9_]*:/, 'variable'],
                
                // Numbers
                [/\b[0-9]+\b/, 'number'],
                
                // Comments
                [/;.*$/, 'comment'],
                
                // Strings
                [/"[^"]*"/, 'string'],
                
                // Brackets and punctuation
                [/[{}()[\]]/, 'delimiter.bracket'],
                [/[,:.]/, 'delimiter']
            ]
        }
    });

    // Common editor options
    const editorOptions = {
        theme: 'vs-dark',
        automaticLayout: true,
        minimap: {
            enabled: true
        },
        scrollBeyondLastLine: false,
        fontSize: 14,
        lineNumbers: 'on',
        roundedSelection: false,
        scrollbar: {
            vertical: 'visible',
            horizontal: 'visible',
            useShadows: false,
            verticalScrollbarSize: 10,
            horizontalScrollbarSize: 10
        },
        readOnly: true
    };

    // Create all four editors
    const originalAsmEditor = monaco.editor.create(document.getElementById('original-asm-container'), {
        ...editorOptions,
        language: 'asm',
        value: '; Original Assembly'
    });

    const predictedAsmEditor = monaco.editor.create(document.getElementById('predicted-asm-container'), {
        ...editorOptions,
        language: 'asm',
        value: '; Predicted Assembly'
    });

    const originalIrEditor = monaco.editor.create(document.getElementById('original-ir-container'), {
        ...editorOptions,
        language: 'llvm',
        value: '; Original LLVM IR'
    });

    const predictedIrEditor = monaco.editor.create(document.getElementById('predicted-ir-container'), {
        ...editorOptions,
        language: 'llvm',
        value: '; Predicted LLVM IR'
    });

    // Create raw message editor
    const rawMessageEditor = monaco.editor.create(document.getElementById('raw-message-container'), {
        ...editorOptions,
        language: 'plaintext',
        value: 'Raw Message Content'
    });

    // Function to load content by index
    function loadContentByIndex(idx) {
        console.log('Loading content for index:', idx);
        fetch('/load_by_index', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ idx: idx })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('Received data:', {
                    original_asm_length: data.original_asm.length,
                    predicted_asm_length: data.predicted_asm.length,
                    original_ir_length: data.original_ir.length,
                    predicted_ir_length: data.predicted_ir.length
                });
                originalAsmEditor.setValue(data.original_asm);
                predictedAsmEditor.setValue(data.predicted_asm);
                originalIrEditor.setValue(data.original_ir);
                predictedIrEditor.setValue(data.predicted_ir);
                rawMessageEditor.setValue(data.raw_message || 'No raw message available');

                // Store the full response data for later use
                window.currentData = data;

                // Update status indicators
                const compileContainer = document.getElementById('compile-status-container');
                const executionContainer = document.getElementById('execution-status-container');

                // Clear previous status indicators
                compileContainer.innerHTML = '';
                executionContainer.innerHTML = '';

                // Update compilation status
                data.predict_compile_success.forEach((success, index) => {
                    const statusItem = document.createElement('div');
                    statusItem.className = 'status-item';
                    statusItem.innerHTML = `
                        <span class="test-number" data-test-idx="${index}">Test ${index + 1}</span>
                        <i class="fas ${success ? 'fa-check' : 'fa-times'}"></i>
                    `;
                    compileContainer.appendChild(statusItem);
                });

                // Update execution status
                data.predict_execution_success.forEach((success, index) => {
                    const statusItem = document.createElement('div');
                    statusItem.className = 'status-item';
                    statusItem.innerHTML = `
                        <span class="test-number" data-test-idx="${index}">Test ${index + 1}</span>
                        <i class="fas ${success ? 'fa-check' : 'fa-times'}"></i>
                    `;
                    executionContainer.appendChild(statusItem);
                });

                // Add click event listeners to all test numbers
                document.querySelectorAll('.test-number').forEach(element => {
                    element.addEventListener('click', function() {
                        const testIdx = parseInt(this.getAttribute('data-test-idx'));
                        updatePredictedContent(testIdx);
                    });
                });
            } else {
                console.error('Error loading content:', data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Function to update predicted content based on test index
    function updatePredictedContent(testIdx) {
        if (!window.currentData) return;

        // First update the predict index on the server
        fetch('/update_predict_idx', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ idx: testIdx })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // After updating the index, reload the content
                const currentIdx = parseInt(document.getElementById('idx-input').value);
                if (!isNaN(currentIdx) && currentIdx >= 0) {
                    loadContentByIndex(currentIdx);
                }
            } else {
                console.error('Error updating predict index:', data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });

        // Highlight the selected test
        document.querySelectorAll('.test-number').forEach(element => {
            const elementTestIdx = parseInt(element.getAttribute('data-test-idx'));
            if (elementTestIdx === testIdx) {
                element.classList.add('selected');
            } else {
                element.classList.remove('selected');
            }
        });
    }

    // Add event listener to the load button
    document.getElementById('load-button').addEventListener('click', () => {
        const idx = parseInt(document.getElementById('idx-input').value);
        if (!isNaN(idx) && idx >= 0) {
            loadContentByIndex(idx);
        }
    });

    // Add event listener to the input box for Enter key
    document.getElementById('idx-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const idx = parseInt(e.target.value);
            if (!isNaN(idx) && idx >= 0) {
                loadContentByIndex(idx);
            }
        }
    });
}); 