import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Settings } from 'lucide-react';

const TrafficLightGA = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [population, setPopulation] = useState([]);
  const [bestSolution, setBestSolution] = useState(null);
  const [fitnessHistory, setFitnessHistory] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  
  // GA Parameters
  const [params, setParams] = useState({
    populationSize: 50,
    mutationRate: 0.15,
    crossoverRate: 0.8,
    elitismCount: 2,
    maxGenerations: 100
  });

  // Traffic light zones with vehicle counts
  const [zones, setZones] = useState([
    { id: 1, name: 'North', vehicles: 45, minTime: 20, maxTime: 90 },
    { id: 2, name: 'South', vehicles: 38, minTime: 20, maxTime: 90 },
    { id: 3, name: 'East', vehicles: 52, minTime: 20, maxTime: 90 },
    { id: 4, name: 'West', vehicles: 30, minTime: 20, maxTime: 90 }
  ]);

  // Create random chromosome (traffic light timings)
  const createChromosome = () => {
    return zones.map(zone => ({
      zoneId: zone.id,
      greenTime: Math.floor(Math.random() * (zone.maxTime - zone.minTime + 1)) + zone.minTime
    }));
  };

  // Calculate fitness score
  const calculateFitness = (chromosome) => {
    let totalWaitTime = 0;
    let totalCycleTime = chromosome.reduce((sum, gene) => sum + gene.greenTime, 0);
    
    // Add yellow and red clearance time (assume 3 seconds per transition)
    totalCycleTime += zones.length * 3;

    chromosome.forEach((gene, idx) => {
      const zone = zones.find(z => z.id === gene.zoneId);
      const vehiclesPerSecond = zone.vehicles / 60; // vehicles per minute to per second
      
      // Calculate how many vehicles can pass during green time
      const vehiclesThroughput = (gene.greenTime / 3) * 1.5; // assume 1.5 vehicles per 3 seconds
      
      // Vehicles that couldn't pass have to wait for full cycle
      const waitingVehicles = Math.max(0, zone.vehicles - vehiclesThroughput);
      
      // Average wait time per vehicle
      const avgWaitTime = waitingVehicles * totalCycleTime / 2;
      
      // Penalty for over-allocation (wasted green time)
      const overallocation = Math.max(0, vehiclesThroughput - zone.vehicles);
      
      totalWaitTime += avgWaitTime + (overallocation * 5);
    });

    // Add penalty for very long or very short cycles
    const cyclePenalty = Math.abs(totalCycleTime - 120) * 0.5;
    
    // Lower wait time = higher fitness
    return 10000 / (totalWaitTime + cyclePenalty + 1);
  };

  // Initialize population
  const initializePopulation = () => {
    const pop = [];
    for (let i = 0; i < params.populationSize; i++) {
      const chromosome = createChromosome();
      pop.push({
        chromosome,
        fitness: calculateFitness(chromosome)
      });
    }
    return pop.sort((a, b) => b.fitness - a.fitness);
  };

  // Tournament selection
  const tournamentSelection = (pop) => {
    const tournamentSize = 3;
    let best = pop[Math.floor(Math.random() * pop.length)];
    
    for (let i = 1; i < tournamentSize; i++) {
      const contestant = pop[Math.floor(Math.random() * pop.length)];
      if (contestant.fitness > best.fitness) {
        best = contestant;
      }
    }
    return best;
  };

  // Crossover
  const crossover = (parent1, parent2) => {
    if (Math.random() > params.crossoverRate) {
      return [parent1.chromosome, parent2.chromosome];
    }

    const point = Math.floor(Math.random() * zones.length);
    const child1 = [
      ...parent1.chromosome.slice(0, point),
      ...parent2.chromosome.slice(point)
    ];
    const child2 = [
      ...parent2.chromosome.slice(0, point),
      ...parent1.chromosome.slice(point)
    ];

    return [child1, child2];
  };

  // Mutation
  const mutate = (chromosome) => {
    return chromosome.map(gene => {
      if (Math.random() < params.mutationRate) {
        const zone = zones.find(z => z.id === gene.zoneId);
        const change = Math.floor(Math.random() * 21) - 10; // -10 to +10
        const newTime = Math.max(zone.minTime, Math.min(zone.maxTime, gene.greenTime + change));
        return { ...gene, greenTime: newTime };
      }
      return gene;
    });
  };

  // Evolve population
  const evolvePopulation = (pop) => {
    const newPop = [];

    // Elitism - keep best solutions
    for (let i = 0; i < params.elitismCount; i++) {
      newPop.push(pop[i]);
    }

    // Create offspring
    while (newPop.length < params.populationSize) {
      const parent1 = tournamentSelection(pop);
      const parent2 = tournamentSelection(pop);
      
      const [child1, child2] = crossover(parent1, parent2);
      
      const mutatedChild1 = mutate(child1);
      const mutatedChild2 = mutate(child2);

      newPop.push({
        chromosome: mutatedChild1,
        fitness: calculateFitness(mutatedChild1)
      });

      if (newPop.length < params.populationSize) {
        newPop.push({
          chromosome: mutatedChild2,
          fitness: calculateFitness(mutatedChild2)
        });
      }
    }

    return newPop.sort((a, b) => b.fitness - a.fitness);
  };

  // Run one generation
  const runGeneration = () => {
    setPopulation(prev => {
      const newPop = evolvePopulation(prev);
      const best = newPop[0];
      
      setBestSolution(best);
      setFitnessHistory(prev => [...prev, best.fitness]);
      setGeneration(g => g + 1);
      
      return newPop;
    });
  };

  // Initialize
  useEffect(() => {
    const initialPop = initializePopulation();
    setPopulation(initialPop);
    setBestSolution(initialPop[0]);
    setFitnessHistory([initialPop[0].fitness]);
  }, []);

  // Auto-run
  useEffect(() => {
    let interval;
    if (isRunning && generation < params.maxGenerations) {
      interval = setInterval(runGeneration, 100);
    } else if (generation >= params.maxGenerations) {
      setIsRunning(false);
    }
    return () => clearInterval(interval);
  }, [isRunning, generation]);

  const reset = () => {
    setIsRunning(false);
    setGeneration(0);
    const initialPop = initializePopulation();
    setPopulation(initialPop);
    setBestSolution(initialPop[0]);
    setFitnessHistory([initialPop[0].fitness]);
  };

  const totalCycleTime = bestSolution 
    ? bestSolution.chromosome.reduce((sum, g) => sum + g.greenTime, 0) + (zones.length * 3)
    : 0;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-gray-50 to-gray-100 min-h-screen">
      <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Traffic Light Optimization using Genetic Algorithm
        </h1>
        <p className="text-gray-600">
          Optimizing green light duration for each zone based on vehicle counts
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Controls */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Controls</h2>
          
          <div className="space-y-4">
            <div className="flex gap-2">
              <button
                onClick={() => setIsRunning(!isRunning)}
                disabled={generation >= params.maxGenerations}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition"
              >
                {isRunning ? <Pause size={20} /> : <Play size={20} />}
                {isRunning ? 'Pause' : 'Start'}
              </button>
              
              <button
                onClick={reset}
                className="flex items-center justify-center gap-2 px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
              >
                <RotateCcw size={20} />
              </button>
            </div>

            <button
              onClick={() => setShowSettings(!showSettings)}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition"
            >
              <Settings size={18} />
              {showSettings ? 'Hide' : 'Show'} Settings
            </button>

            {showSettings && (
              <div className="space-y-3 pt-3 border-t">
                {Object.entries(params).map(([key, value]) => (
                  <div key={key}>
                    <label className="block text-sm font-medium text-gray-700 mb-1 capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </label>
                    <input
                      type="number"
                      value={value}
                      onChange={(e) => setParams(prev => ({...prev, [key]: parseFloat(e.target.value)}))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                      step={key.includes('Rate') ? 0.05 : 1}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <div className="text-sm text-gray-600 mb-1">Generation</div>
            <div className="text-3xl font-bold text-blue-600">
              {generation} / {params.maxGenerations}
            </div>
            <div className="mt-3 text-sm text-gray-600">Best Fitness</div>
            <div className="text-2xl font-bold text-green-600">
              {bestSolution ? bestSolution.fitness.toFixed(2) : '0'}
            </div>
          </div>
        </div>

        {/* Traffic Zones Input */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Traffic Zones</h2>
          
          <div className="space-y-4">
            {zones.map((zone, idx) => (
              <div key={zone.id} className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-gray-800">{zone.name}</span>
                  <span className="text-sm text-gray-600">Zone {zone.id}</span>
                </div>
                
                <label className="block text-sm text-gray-600 mb-1">
                  Vehicle Count
                </label>
                <input
                  type="number"
                  value={zone.vehicles}
                  onChange={(e) => {
                    const newZones = [...zones];
                    newZones[idx].vehicles = parseInt(e.target.value) || 0;
                    setZones(newZones);
                    reset();
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Best Solution */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Optimized Timings</h2>
          
          {bestSolution && (
            <div className="space-y-4">
              {bestSolution.chromosome.map((gene) => {
                const zone = zones.find(z => z.id === gene.zoneId);
                const percentage = (gene.greenTime / totalCycleTime) * 100;
                
                return (
                  <div key={gene.zoneId} className="p-4 bg-gradient-to-r from-green-50 to-green-100 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-gray-800">{zone.name}</span>
                      <span className="text-sm text-gray-600">{zone.vehicles} vehicles</span>
                    </div>
                    
                    <div className="flex items-center gap-3">
                      <div className="flex-1">
                        <div className="h-6 bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-green-600 transition-all duration-300"
                            style={{width: `${percentage}%`}}
                          />
                        </div>
                      </div>
                      <span className="text-2xl font-bold text-green-700 w-16 text-right">
                        {gene.greenTime}s
                      </span>
                    </div>
                    
                    <div className="mt-2 text-xs text-gray-600">
                      {percentage.toFixed(1)}% of cycle time
                    </div>
                  </div>
                );
              })}
              
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <div className="text-sm text-gray-600">Total Cycle Time</div>
                <div className="text-2xl font-bold text-blue-600">
                  {totalCycleTime} seconds
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  (includes 3s transition per zone)
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Fitness Chart */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Fitness Evolution</h2>
        
        <div className="relative h-64 bg-gray-50 rounded-lg p-4">
          <svg className="w-full h-full">
            {fitnessHistory.length > 1 && (
              <polyline
                points={fitnessHistory.map((fitness, idx) => {
                  const x = (idx / (fitnessHistory.length - 1)) * 100;
                  const maxFitness = Math.max(...fitnessHistory);
                  const minFitness = Math.min(...fitnessHistory);
                  const range = maxFitness - minFitness || 1;
                  const y = 100 - ((fitness - minFitness) / range) * 80;
                  return `${x}%,${y}%`;
                }).join(' ')}
                fill="none"
                stroke="#3b82f6"
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
            )}
          </svg>
          
          <div className="absolute bottom-2 left-4 text-xs text-gray-500">
            Gen 0
          </div>
          <div className="absolute bottom-2 right-4 text-xs text-gray-500">
            Gen {generation}
          </div>
        </div>
      </div>

      {/* Algorithm Explanation */}
      <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">How It Works</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-700">
          <div>
            <h3 className="font-semibold text-gray-800 mb-2">1. Chromosome Encoding</h3>
            <p>Each solution is a set of green light durations for all zones.</p>
          </div>
          
          <div>
            <h3 className="font-semibold text-gray-800 mb-2">2. Fitness Function</h3>
            <p>Minimizes total vehicle wait time while balancing throughput and cycle efficiency.</p>
          </div>
          
          <div>
            <h3 className="font-semibold text-gray-800 mb-2">3. Selection</h3>
            <p>Tournament selection picks better solutions with higher probability.</p>
          </div>
          
          <div>
            <h3 className="font-semibold text-gray-800 mb-2">4. Evolution</h3>
            <p>Crossover combines solutions, mutation explores variations, elitism preserves best.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrafficLightGA;