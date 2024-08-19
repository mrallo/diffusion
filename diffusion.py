import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random

class Diffusion:
    def __init__(self, size, temperature):
        """
        Initialize a diffusion process.

        Args:
            size (int): The size of the chamber (width and height).
            temperature (int): The temperature of the chamber.
        """
         # Fill up the 80% of the chamber
        self.chamber = Chamber(size, temperature, int(0.8 * size * size))

        # Initialize the display backend
        self.init_display()

        # Hook up to the diffusion chamber a couple of useful probes:
        # 1. Display the concentration on the scope screen
        ConcentrationDensity(self.chamber, self.scope_screen)
        # 2. Entropy measures the disorder used to determine when the two substances are properly mixed
        SystemEntropy(self.chamber, self.scope_screen)

    def init_display(self):
        self.fig, (self.ax, self.scope_screen) = plt.subplots(1, 2, figsize=(16, 8))
        self.ax.set_xlim(0, self.chamber.width)
        self.ax.set_ylim(0, self.chamber.height)
        self.ax.set_xlabel('Width')
        self.ax.set_ylabel('Height')
        self.ax.set_title(f'Diffusion Chamber Temperature: {self.chamber.temperature}K')
        self.scope_screen.set_xlim(0, self.chamber.width)
        self.scope_screen.set_ylim(0, self.chamber.height)
        self.scat = self.ax.scatter([p.x for p in self.chamber.particles], [p.y for p in self.chamber.particles], c=[p.color for p in self.chamber.particles], s=10)

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=None, interval=30, cache_frame_data=False, repeat=True)
        plt.show()

    def update(self, frame):
        self.chamber.update(frame)
        self.scat.set_offsets([(p.x, p.y) for p in self.chamber.particles])
        return self.scat

class Chamber:
    def __init__(self, size, temperature, num_particles):
        """
        Initialize a Chamber object.

        Args:
            size (int): The size of the chamber (width and height).
            temperature (int): The temperature of the chamber.
            num_particles (int): The number of particles in the chamber.
        """
        self.width = self.height = size
        self.temperature = temperature
        self.particles = []
        self.hooks = []
        self.init_particles(num_particles)

    def init_particles(self, num_particles):
        """
        Initialize the particles in the chamber.

        Args:
            num_particles (int): The number of particles to initialize.
        """
        mid_x = self.width // 2
        for _ in range(num_particles):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            # Avoid particle collisions
            while self.is_position_occupied(x, y):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
            self.particles.append(Particle(x, y, 'orange' if x < mid_x else 'blue'))

    def add_hook(self, hook):
        self.hooks.append(hook)

    def is_position_occupied(self, x, y):
        """
        Check if a position in the chamber is occupied by a particle.

        Args:
            x (int): The x-coordinate of the position.
            y (int): The y-coordinate of the position.

        Returns:
            bool: True if the position is occupied, False otherwise.
        """
        for particle in self.particles:
            if particle.x == x and particle.y == y:
                return True
        return False

    def update(self, cycle):
        """
        Update the particle positions and call the hooks.
        """
        for particle in self.particles:
            particle.move(self.width, self.height, self.temperature / 100)

        for hook in self.hooks:
            hook.updated(cycle)

class Particle:
    def __init__(self, x, y, color):
        """
        Initialize a Particle object.

        Args:
            x (int): The initial x-coordinate of the particle.
            y (int): The initial y-coordinate of the particle.
            color (str): The color of the particle.
        """
        self.x = x
        self.y = y
        self.color = color

    def move(self, max_x, max_y, velocity):
        """
        Move the particle to a new position.

        Args:
            max_x (int): The maximum x-coordinate of the chamber.
            max_y (int): The maximum y-coordinate of the chamber.
            velocity (float): The velocity of the particle.

        Returns:
            tuple: The new x and y coordinates of the particle.
        """
        new_x = (self.x + random.choice([-1, 0, 1]) * velocity)
        new_y = (self.y + random.choice([-1, 0, 1]) * velocity)
        if 0 <= new_x < max_x:
            self.x = new_x
        if 0 <= new_y < max_y:
            self.y = new_y

class Probe:
    def __init__(self, chamber, scope_screen):
        """
        Initialize a Probe object.

        Args:
            chamber (Chamber): The Chamber object to probe.
            scope_screen (matplotlib.axes.Axes): The scope screen to display the probe results.
        """
        self.chamber = chamber
        self.scope_screen = scope_screen
        self.chamber.add_hook(self)

    def updated(self, cycle):
        """
        Callback method called by the Chamber object's update method.

        Args:
            frame (int): The current frame of the animation.
        """
        pass

class ConcentrationDensity(Probe):
    def __init__(self, chamber, scope_screen):
        """
        Initialize a ConcentrationDensity object.

        Args:
            chamber (Chamber): The Chamber object to probe.
            scope_screen (matplotlib.axes.Axes): The scope screen to display the probe results.
        """
        super().__init__(chamber, scope_screen)

    def updated(self, cycle):
        """
        Callback method called by the Chamber object's update method.

        Args:
            frame (int): The current frame of the animation.
        """
        orange_density = np.zeros((self.chamber.width, self.chamber.height))
        blue_density = np.zeros((self.chamber.width, self.chamber.height))

        for particle in self.chamber.particles:
            if particle.color == 'orange':
                orange_density[int(particle.x), int(particle.y)] += 1
            elif particle.color == 'blue':
                blue_density[int(particle.x), int(particle.y)] += 1

        x_values = np.arange(self.chamber.width)

        orange_concentration = np.sum(orange_density, axis=1)
        blue_concentration = np.sum(blue_density, axis=1)

        # Plot the updated concentrations
        self.scope_screen.clear()
        self.scope_screen.plot(x_values, orange_concentration, color='orange', label='Orange Concentration')
        self.scope_screen.plot(x_values, blue_concentration, color='blue', label='Blue Concentration')
        self.scope_screen.set_xlabel('X')
        self.scope_screen.set_ylabel('Concentration')
        self.scope_screen.legend(loc='upper right')

class SystemEntropy(Probe):
    def __init__(self, chamber, scope_screen):
        """
        Initialize a SystemEntropy object.

        Args:
            chamber (Chamber): The Chamber object to probe.
            scope_screen (matplotlib.axes.Axes): The scope screen to display the probe results.
        """
        super().__init__(chamber, scope_screen)
        self.max_entropy = 0
        self.max_entropy_cycle = 0

    def updated(self, cycle):
        """
        Callback method called by the Chamber object's update method.

        Args:
            frame (int): The current frame of the animation.
        """
        orange_density = np.zeros((self.chamber.width, self.chamber.height))
        blue_density = np.zeros((self.chamber.width, self.chamber.height))

        for particle in self.chamber.particles:
            if particle.color == 'orange':
                orange_density[int(particle.x), int(particle.y)] += 1
            elif particle.color == 'blue':
                blue_density[int(particle.x), int(particle.y)] += 1

        # Calculate entropy
        total_particles = orange_density + blue_density
        total_particles[total_particles == 0] = 1 # Avoid division by zero

        # orange_prob and blue_prob are the probabilities of finding a orange or blue particle in each cell
        orange_prob = orange_density / total_particles
        blue_prob = blue_density / total_particles

        # Calculate entropy for each cell. Add a small value to avoid log(0)
        entropy = - (orange_prob * np.log2(orange_prob + 1e-9) + blue_prob * np.log2(blue_prob + 1e-9))
        total_entropy = np.sum(entropy)

        # Update and display system entropy
        if total_entropy > self.max_entropy:
            self.max_entropy = total_entropy
            self.max_entropy_cycle = cycle
        self.scope_screen.set_title(f'Cycle: {cycle}, Entropy: {total_entropy:.2f} (Max Entropy: {self.max_entropy:.2f} at Cycle: {self.max_entropy_cycle})')

if __name__ == "__main__":
    size = 100
    temperature = 75
    diffusion = Diffusion(size, temperature)
    diffusion.start()