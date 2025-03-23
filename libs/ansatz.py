from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import QuantumCircuit, Parameter

class Ansatz():
    def __init__(self, ansatz_dict, latent_bit_num, trash_bit_num):
        self.num_qubits = latent_bit_num + trash_bit_num
        self.latent_bit_num = latent_bit_num
        self.reps = ansatz_dict['ansatz_reps']
        
        A = ansatz_dict['A']
        A_gate = ansatz_dict['A_gate']
        B = ansatz_dict['B']
        B_gate = ansatz_dict['B_gate']
        
        self.ansatz = self.proposed_ansatz(A, A_gate, B, B_gate)

        
    def proposed_ansatz(self, A, A_gate, B, B_gate):
        
        qc = QuantumCircuit(self.num_qubits)
        
        # A
        if A == 'A-1':
            for i in range(self.num_qubits):
                if A_gate == 'Rx':
                    qc.rx(Parameter(f"θ{len(qc.parameters):04d}"), i)
                elif A_gate == 'Ry':
                    qc.ry(Parameter(f"θ{len(qc.parameters):04d}"), i)
                elif A_gate == 'Rz':
                    qc.rz(Parameter(f"θ{len(qc.parameters):04d}"), i)
                
        # B
        for _ in range(self.reps):

            if B == 'B-1':
                for i in range(self.num_qubits - 1, 0, -1):
                    qc.cx(i-1, i)
                qc.barrier()
                for i in range(self.num_qubits):
                    if B_gate == 'Rx':
                        qc.rx(Parameter(f"θ{len(qc.parameters):04d}"), i)
                    elif B_gate == 'Ry':
                        qc.ry(Parameter(f"θ{len(qc.parameters):04d}"), i)
                    elif B_gate == 'Rz':
                        qc.rz(Parameter(f"θ{len(qc.parameters):04d}"), i)
                qc.barrier()

            elif B == 'B-2':
                for i in range(0, self.num_qubits - 1):
                    qc.cx(i, i+1)
                qc.barrier()
                for i in range(self.num_qubits):
                    if B_gate == 'Rx':
                        qc.rx(Parameter(f"θ{len(qc.parameters):04d}"), i)
                    elif B_gate == 'Ry':
                        qc.ry(Parameter(f"θ{len(qc.parameters):04d}"), i)
                    elif B_gate == 'Rz':
                        qc.rz(Parameter(f"θ{len(qc.parameters):04d}"), i)
                qc.barrier()
            
            elif B == 'B-3':
                qc.cx(self.num_qubits - 1, 0)
                for i in range(0, self.num_qubits - 1):
                    qc.cx(i, i+1)
                qc.barrier()
                for i in range(self.num_qubits):
                    if B_gate == 'Rx':
                        qc.rx(Parameter(f"θ{len(qc.parameters):04d}"), i)
                    elif B_gate == 'Ry':
                        qc.ry(Parameter(f"θ{len(qc.parameters):04d}"), i)
                    elif B_gate == 'Rz':
                        qc.rz(Parameter(f"θ{len(qc.parameters):04d}"), i)
                qc.barrier()

            elif B == 'B-4':
                for x in range(self.num_qubits):
                    for y in range(x + 1, self.num_qubits):
                        if (x%2 == 0 and y%2 == 0) or (x%2 == 1 and y%2 == 1):
                            if B_gate == 'Rx':
                                qc.crx(Parameter(f"θ{len(qc.parameters):04d}"), y, x)
                            elif B_gate == 'Ry':
                                qc.cry(Parameter(f"θ{len(qc.parameters):04d}"), y, x)
                        else:
                            if B_gate == 'Rx':
                                qc.crx(Parameter(f"θ{len(qc.parameters):04d}"), x, y)
                            elif B_gate == 'Ry':
                                qc.cry(Parameter(f"θ{len(qc.parameters):04d}"), x, y)    
                    qc.barrier()
            
            elif B == 'B-5':                
                for x in range(self.num_qubits):
                    qc.rx(Parameter(f"θ{len(qc.parameters):04d}"), x)
                
                for x in range(self.num_qubits):
                    for y in range(self.num_qubits):
                        if x != y: 
                            if B_gate == 'Rx':
                                qc.crx(Parameter(f"θ{len(qc.parameters):04d}"), x, y)
                            elif B_gate == 'Ry':
                                qc.cry(Parameter(f"θ{len(qc.parameters):04d}"), x, y)                    
                    qc.barrier()   
                    
                for x in range(self.num_qubits):
                    qc.rx(Parameter(f"θ{len(qc.parameters):04d}"), x)                
        
        return qc