module QuantumCircuitsMPSLuxorExt

using Luxor
using QuantumCircuitsMPS
using QuantumCircuitsMPS: Circuit, expand_circuit, ExpandedOp

"""Wrapper type for SVG data that auto-displays in Jupyter notebooks."""
struct SVGImage
    data::String
end

# MIME display method for IJulia auto-rendering
function Base.show(io::IO, ::MIME"image/svg+xml", img::SVGImage)
    write(io, img.data)
end

"""
    plot_circuit(circuit::Circuit; seed::Int=0, filename::Union{String, Nothing}=nothing)

Export a quantum circuit diagram to SVG using Luxor.jl.

Renders the circuit as a wire diagram with:
- Vertical lines representing qubit wires (labeled q1, q2, ...)
- Boxes with gate labels at sites where gates act
- Row headers showing step numbers (with letter suffixes for multi-op steps)
- Time axis goes upward, qubits spread horizontally

# Arguments
- `circuit::Circuit`: The circuit to visualize
- `seed::Int=0`: RNG seed for stochastic branch resolution (same seed = same diagram)
- `filename::Union{String, Nothing}=nothing`: Output file path (SVG format). If `nothing`, returns `SVGImage` for auto-display in Jupyter.

# Returns
- If `filename === nothing`: Returns `SVGImage` wrapper (auto-displays in Jupyter notebooks)
- If `filename` provided: Writes to file and returns `nothing`

# Requirements
Requires `Luxor` to be loaded (`using Luxor` before calling).

# Example
```julia
using QuantumCircuitsMPS
using Luxor  # Load the extension

circuit = Circuit(L=4, bc=:periodic, n_steps=5) do c
    apply!(c, Reset(), StaircaseRight(1))
    apply_with_prob!(c; rng=:ctrl, outcomes=[
        (probability=0.5, gate=HaarRandom(), geometry=StaircaseLeft(4))
    ])
end

# Auto-display in Jupyter (returns SVGImage)
plot_circuit(circuit; seed=42)

# Export to file
plot_circuit(circuit; seed=42, filename="my_circuit.svg")
```

# Determinism
Using the same `seed` value produces identical diagrams. The seed controls
which stochastic branches are displayed, matching the behavior of
`expand_circuit(circuit; seed=seed)`.

# See Also
- [`print_circuit`](@ref): ASCII visualization (no Luxor required)
- [`expand_circuit`](@ref): Get the concrete operations being visualized
"""
function QuantumCircuitsMPS.plot_circuit(circuit::Circuit; seed::Int=0, filename::Union{String, Nothing}=nothing)
    # TODO: Known bug - non-adjacent gates (e.g., NNN gates) are not rendered correctly.
    # The current implementation assumes gates act on adjacent or contiguous qubit ranges.
    
    # Layout constants
    QUBIT_SPACING = 40.0
    ROW_HEIGHT = 60.0  # Height per time step (was COLUMN_WIDTH)
    GATE_WIDTH = 30.0   # Width along qubit axis
    GATE_HEIGHT = 40.0  # Height along time axis
    MARGIN = 50.0
    MIN_FONT_SIZE = 8.0  # Minimum font size before truncation
    DEFAULT_FONT_SIZE = 11.0  # Default Luxor font size
    
    # Helper: Calculate font size to fit text in box, with truncation fallback
    function calc_font_size(label::String, box_width::Float64, default_size::Float64=DEFAULT_FONT_SIZE)
        # Set default font size to measure
        fontsize(default_size)
        extents = textextents(label)
        text_width = extents[3]  # width is 3rd element
        
        # If fits at default size, return default
        if text_width <= box_width * 0.9  # 90% of box for padding
            return (default_size, label)
        end
        
        # Scale down proportionally
        scale_factor = (box_width * 0.9) / text_width
        scaled_size = default_size * scale_factor
        
        # If scaled size >= minimum, use it
        if scaled_size >= MIN_FONT_SIZE
            return (scaled_size, label)
        end
        
        # At minimum size, check if truncation needed
        fontsize(MIN_FONT_SIZE)
        extents = textextents(label)
        text_width = extents[3]
        
        if text_width <= box_width * 0.9
            return (MIN_FONT_SIZE, label)
        end
        
        # Truncate with "..."
        truncated = label
        extents = textextents(truncated * "...")
        while extents[3] > box_width * 0.9 && length(truncated) > 1
            truncated = truncated[1:end-1]
            extents = textextents(truncated * "...")
        end
        
        return (MIN_FONT_SIZE, truncated * "...")
    end
    
    # Helper: Draw a filled white box with black stroke at (x, y) position
    function render_gate_box(x, y, width, height)
        setcolor("white")
        box(Point(x, y), width, height, :fill)
        setcolor("black")
        box(Point(x, y), width, height, :stroke)
    end
    
    # Helper: Render label with dynamic font sizing, centered at (x, y)
    function render_gate_label(x, y, label, max_width)
        (font_sz, display_label) = calc_font_size(label, max_width)
        fontsize(font_sz)
        text(display_label, Point(x, y + 5), halign=:center, valign=:center)
    end
    
    # Helper: Draw connecting line between two points (optional dashed style)
    function render_connecting_line(pt1, pt2; dashed=false)
        if dashed
            setdash("dashed")
        end
        line(pt1, pt2, :stroke)
        if dashed
            setdash("solid")  # Reset to solid
        end
    end
    
    # Expand circuit to get concrete operations
    expanded = expand_circuit(circuit; seed=seed)
    
    # Helper: check if two ops overlap (share any qubits)
    function ops_overlap(op1, op2)
        return !isempty(intersect(op1.sites, op2.sites))
    end
    
    # Helper: check if any ops in the list overlap with each other
    function any_ops_overlap(ops)
        for i in 1:length(ops), j in (i+1):length(ops)
            if ops_overlap(ops[i], ops[j])
                return true
            end
        end
        return false
    end
    
    # Build row list with visual row position tracking
    # Each row is: (step_idx, letter, op, row_pos)
    # row_pos is the visual row index (1-based) for Y coordinate calculation
    rows = []
    visual_row = 0  # Track current visual row position
    for (step_idx, step_ops) in enumerate(expanded)
        if isempty(step_ops)
            # Empty step - still render one row
            visual_row += 1
            push!(rows, (step_idx, "", nothing, visual_row))
        elseif length(step_ops) == 1
            # Single op - no letter suffix
            visual_row += 1
            push!(rows, (step_idx, "", step_ops[1], visual_row))
        else
            # Multiple ops - check for overlaps
            if any_ops_overlap(step_ops)
                # Overlapping ops: staggered layout with letter suffixes
                for (substep_idx, op) in enumerate(step_ops)
                    visual_row += 1
                    letter = Char('a' + substep_idx - 1)
                    push!(rows, (step_idx, string(letter), op, visual_row))
                end
            else
                # Non-overlapping ops: parallel layout, same visual row, no letter suffixes
                visual_row += 1
                for op in step_ops
                    push!(rows, (step_idx, "", op, visual_row))
                end
            end
        end
    end
    n_visual_rows = visual_row
    
    # Calculate canvas size (swapped dimensions)
    canvas_width = 2 * MARGIN + circuit.L * QUBIT_SPACING + 100  # qubit dimension
    canvas_height = 2 * MARGIN + n_visual_rows * ROW_HEIGHT      # time dimension
    
    # Conditional: in-memory mode vs file mode
    if filename === nothing
        # In-memory mode: create SVG drawing surface
        Drawing(canvas_width, canvas_height, :svg)
    else
        # File mode: create drawing with filename
        Drawing(canvas_width, canvas_height, filename)
    end
    
    background("white")
    origin(Point(MARGIN, MARGIN))
    
    # Draw vertical qubit wires (was horizontal)
    wire_length = n_visual_rows * ROW_HEIGHT
    for q in 1:circuit.L
        x = q * QUBIT_SPACING  # was y
        line(Point(x, 0), Point(x, wire_length), :stroke)
        # Qubit label at bottom
        text("q$q", Point(x, wire_length + 20), halign=:center)
    end
    
    # Draw step headers on left side (was top)
    # For parallel ops (same row_pos), only render header once
    rendered_headers = Set{Int}()
    for (step, letter, _, row_pos) in rows
        if row_pos ∉ rendered_headers
            push!(rendered_headers, row_pos)
            y = wire_length - (row_pos - 0.5) * ROW_HEIGHT  # was x
            header = letter == "" ? string(step) : "$(step)$(letter)"
            text(header, Point(-10, y + 5), halign=:right, valign=:center)
        end
    end
    
    # Draw gate boxes with transposed coordinates
    for (_, _, op, row_pos) in rows
        if op !== nothing
            y = wire_length - (row_pos - 0.5) * ROW_HEIGHT  # time position (was x)
            
            # Check if single-qubit or multi-qubit gate
            if length(op.sites) == 1
                # Single-qubit gate
                x = op.sites[1] * QUBIT_SPACING  # qubit position (was y)
                
                render_gate_box(x, y, GATE_WIDTH, GATE_HEIGHT)
                render_gate_label(x, y, op.label, GATE_WIDTH)
            else
                # Multi-qubit gate
                min_site = minimum(op.sites)
                max_site = maximum(op.sites)
                span = max_site - min_site
                L = circuit.L
                
                # Calculate wrapped span (for periodic BC, adjacent pairs like [8,1] have span=L-1 not 1)
                # wrapped_span is the "short way around" for periodic BC
                wrapped_span = min(span, L - span)
                
                # Determine rendering mode:
                # - Adjacent: wrapped_span == 1 (includes periodic wraps like [8,1] → [1,8] → span=7, L-span=1)
                # - Spanning all qubits: length == L (render as single box)
                # - Non-adjacent: wrapped_span > 1 (NNN or larger gaps)
                is_adjacent = (wrapped_span == 1)
                spans_all = (length(op.sites) == L)
                
                if is_adjacent || spans_all
                    # Adjacent or all-spanning gate: render single spanning box
                    # For adjacent wrapping gates like [8,1], center at wrap point
                    if span == L - 1  # This is a wrapping adjacent pair
                        # Render at the boundary - put box between max_site and 1
                        # Use max_site position (right edge of chain)
                        center_x = ((max_site + (L + 1)) / 2) * QUBIT_SPACING
                        span_width = GATE_WIDTH + QUBIT_SPACING  # Same width as adjacent pair
                        # Clamp to stay within canvas
                        center_x = max_site * QUBIT_SPACING  # Just center on max_site for simplicity
                    else
                        center_x = ((min_site + max_site) / 2) * QUBIT_SPACING
                        span_width = span * QUBIT_SPACING + GATE_WIDTH
                    end
                    
                    render_gate_box(center_x, y, span_width, GATE_HEIGHT)
                    render_gate_label(center_x, y, op.label, span_width)
                else
                    # Non-adjacent gate: render two boxes + connecting line
                    x_min = min_site * QUBIT_SPACING
                    x_max = max_site * QUBIT_SPACING
                    
                    # Draw boxes at both sites
                    render_gate_box(x_min, y, GATE_WIDTH, GATE_HEIGHT)
                    render_gate_box(x_max, y, GATE_WIDTH, GATE_HEIGHT)
                    
                    # Determine line style: dashed for periodic wrapping, solid for NNN
                    # Periodic wrapping: span > L/2 (the "long way" around the chain)
                    # e.g., [1, 7] on L=8: span=6 > 4, so it wraps (actual distance is 2 via boundary)
                    # NNN like [1, 3]: span=2 <= 4, so not wrapping
                    is_periodic_wrap = (span > L / 2)
                    
                    # Draw connecting line between boxes (from right edge of left box to left edge of right box)
                    line_start_x = x_min + GATE_WIDTH / 2
                    line_end_x = x_max - GATE_WIDTH / 2
                    render_connecting_line(Point(line_start_x, y), Point(line_end_x, y); dashed=is_periodic_wrap)
                    
                    # Draw label centered between the two boxes
                    center_x = (x_min + x_max) / 2
                    label_width = x_max - x_min  # Distance between box centers
                    (font_sz, display_label) = calc_font_size(op.label, label_width)
                    fontsize(font_sz)
                    
                    # Draw white background for label (to make it readable over the line)
                    extents = textextents(display_label)
                    label_bg_width = extents[3] + 6
                    label_bg_height = extents[4] + 4
                    setcolor("white")
                    box(Point(center_x, y), label_bg_width, label_bg_height, :fill)
                    setcolor("black")
                    text(display_label, Point(center_x, y + 5), halign=:center, valign=:center)
                end
            end
        end
    end
    
    finish()
    
    # Return appropriate value based on mode
    if filename === nothing
        # In-memory mode: extract SVG and return wrapper
        return SVGImage(svgstring())
    else
        # File mode: return nothing (backward compatibility)
        return nothing
    end
end

end # module
