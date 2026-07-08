import pyvista as pv
import numpy as np
from collections import defaultdict


class GeometryViewer:

    def __init__(self, mesh):
        self.mesh = mesh.copy()
        self.contour_actor = None
        self.plotter = None
        self.active_point = None
        self.active_contour = None
        self.adj = None

        # Numbering and threshold state
        self.current_tooth_number = 1
        self.labeled_teeth = []  # list of labeled tooth history dicts
        self.vmax = 1.0
        self.T_green = 0.26
        self.max_radius = 8.0  # Default maximum BFS radius in mm

    def _update_contour(self):
        if self.active_point is None or self.adj is None:
            return

        # Remove previous temporary red contour
        if self.contour_actor is not None:
            self.plotter.remove_actor(self.contour_actor)
            self.contour_actor = None
            self.active_contour = None

        # Find closest vertex index to the click point
        pid = self.mesh.find_closest_point(self.active_point)

        # BFS region growing with physical distance limit (max_radius)
        visited = set()
        queue = [pid]
        visited.add(pid)

        while queue:
            curr = queue.pop(0)
            
            # Check physical distance from the starting click point
            dist = np.linalg.norm(self.mesh.points[curr] - self.active_point)
            if dist >= self.max_radius:
                continue

            for nb in self.adj[curr]:
                if nb not in visited:
                    # Grow only into vertices whose curvature is below the threshold T_green
                    if self.mesh["values"][nb] < self.T_green:
                        visited.add(nb)
                        queue.append(nb)
                    else:
                        # Stop growth at the boundary but include it in the region to form the loop
                        visited.add(nb)

        try:
            if len(visited) > 0:
                mask = np.zeros(self.mesh.n_points, dtype=bool)
                mask[list(visited)] = True

                # Extract subgrid of the grown tooth region
                subgrid = self.mesh.extract_points(mask, adjacent_cells=False)
                
                # Convert UnstructuredGrid to PolyData surface
                surface = subgrid.extract_surface()
                
                # Extract boundary edges of the surface patch (guarantees a closed loop)
                boundary = surface.extract_feature_edges(
                    boundary_edges=True,
                    non_manifold_edges=False,
                    feature_edges=False,
                    manifold_edges=False,
                )

                if boundary.n_points > 0:
                    try:
                        # Extract the main boundary loop closest to the clicked point
                        connected = boundary.connectivity(
                            extraction_mode='closest',
                            closest_point=self.active_point
                        )
                    except Exception:
                        connected = boundary

                    self.active_contour = connected
                    self.contour_actor = self.plotter.add_mesh(
                        connected,
                        color="red",
                        line_width=6,
                        render_lines_as_tubes=True,
                    )
        except Exception as e:
            print("Failed to compute region boundary:", e)

        self.plotter.render()

    def show(self, values, title="Normal Variation"):
        # Make a copy and compute normals for point projection
        mesh_copy = self.mesh.copy()
        mesh_copy.compute_normals(inplace=True, cell_normals=False, point_normals=True)
        self.mesh = mesh_copy.copy()  # Save copy with Normals and values array
        self.mesh["values"] = values  # Keep reference for picking
        
        self.original_values = values
        self.vmax = np.percentile(values, 98)
        self.T_green = 0.26 * self.vmax  # Initial threshold matching lighter green/cyan color

        # Build adjacency graph
        print("Building mesh adjacency for region growing...")
        faces = self.mesh.faces.reshape(-1, 4)[:, 1:]
        self.adj = defaultdict(set)
        for a, b, c in faces:
            self.adj[a].update((b, c))
            self.adj[b].update((a, c))
            self.adj[c].update((a, b))

        # Clip values to self.T_green for flat-color visualization above threshold
        self.mesh_copy = mesh_copy
        self.mesh_copy["values"] = np.minimum(values, self.T_green)

        self.plotter = pv.Plotter()
        self.plotter.add_mesh(
            self.mesh_copy,
            scalars="values",
            cmap="turbo",
            clim=[0, self.vmax],
            smooth_shading=True,
            show_edges=False,
            scalar_bar_args={"title": title},
        )

        def pick_callback(point):
            # Save coordinates of the clicked point
            self.active_point = point
            # Perform region growing and find the closed loop boundary
            self._update_contour()

        def accept_label_callback():
            if self.active_contour is None or self.active_point is None:
                print("No active boundary selected! Click inside a tooth first.")
                return

            num = self.current_tooth_number

            # Render the accepted boundary as a permanent green line
            perm_actor = self.plotter.add_mesh(
                self.active_contour,
                color="green",
                line_width=4,
                render_lines_as_tubes=True,
            )

            # Place a 3D label at the click point
            label_actor = self.plotter.add_point_labels(
                [self.active_point],
                [f"Tooth {num}"],
                font_size=18,
                text_color="white",
                point_color="red",
                point_size=10,
                always_visible=True,
            )

            # Save to labeling history
            self.labeled_teeth.append({
                "contour_actor": perm_actor,
                "label_actor": label_actor,
                "number": num,
            })

            print(f"✓ Labeled Tooth {num} successfully.")
            self.current_tooth_number += 1

            # Reset active selection
            if self.contour_actor is not None:
                self.plotter.remove_actor(self.contour_actor)
                self.contour_actor = None
            self.active_point = None
            self.active_contour = None

            self.plotter.render()

        def undo_callback():
            if not self.labeled_teeth:
                print("No labeled teeth to undo.")
                return

            last_tooth = self.labeled_teeth.pop()
            self.plotter.remove_actor(last_tooth["contour_actor"])
            self.plotter.remove_actor(last_tooth["label_actor"])
            self.current_tooth_number = last_tooth["number"]
            print(f"Undo: Removed label for Tooth {self.current_tooth_number}.")
            self.plotter.render()

        def increase_threshold():
            self.T_green += 0.02 * self.vmax
            print(f"Threshold increased to: {self.T_green:.6f} ({self.T_green/self.vmax*100:.1f}% of vmax)")
            self.plotter.update_scalars(np.minimum(self.original_values, self.T_green), mesh=self.mesh_copy)
            self._update_contour()

        def decrease_threshold():
            self.T_green -= 0.02 * self.vmax
            print(f"Threshold decreased to: {self.T_green:.6f} ({self.T_green/self.vmax*100:.1f}% of vmax)")
            self.plotter.update_scalars(np.minimum(self.original_values, self.T_green), mesh=self.mesh_copy)
            self._update_contour()

        def increase_radius():
            self.max_radius += 0.5
            print(f"Max BFS radius increased to: {self.max_radius:.1f} mm")
            self._update_contour()

        def decrease_radius():
            self.max_radius = max(1.0, self.max_radius - 0.5)
            print(f"Max BFS radius decreased to: {self.max_radius:.1f} mm")
            self._update_contour()

        # Setup interaction callbacks
        self.plotter.enable_point_picking(
            callback=pick_callback,
            left_clicking=True,
            show_point=False,
            picker='cell',  # Use cell picker to prevent picking behind the mesh
        )

        # Bind shortcut keys
        self.plotter.add_key_event("n", accept_label_callback)
        self.plotter.add_key_event("u", undo_callback)
        self.plotter.add_key_event("Up", increase_threshold)
        self.plotter.add_key_event("Down", decrease_threshold)
        self.plotter.add_key_event("Right", increase_radius)
        self.plotter.add_key_event("Left", decrease_radius)

        self.plotter.add_axes()

        print("\n" + "=" * 60)
        print("INTERACTIVE TOOTH NUMBERING CONTROLS:")
        print("  1. Left-click inside a tooth (the clicked point will act as the center).")
        print("     (It will run a BFS and snap a red closed boundary around it).")
        print("  2. Use [Up / Down Arrows] to change green curvature threshold.")
        print("  3. Use [Left / Right Arrows] to shrink / expand the maximum search radius (mm).")
        print("  4. Press [N] to save the boundary and number the tooth.")
        print("  5. Press [U] to undo the last numbered tooth.")
        print("=" * 60 + "\n")

        self.plotter.show()
