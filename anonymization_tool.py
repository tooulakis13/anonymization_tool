import sys
import pandas as pd
import numpy as np
from scipy import stats
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QSpinBox, 
                             QDoubleSpinBox, QTextEdit, QGroupBox, QTabWidget, QProgressBar,
                             QToolTip)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QCursor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph as pg
import time

class AnonymizationThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(object, object)

    def __init__(self, anonymizer, method, params):
        super().__init__()
        self.anonymizer = anonymizer
        self.method = method
        self.params = params

    def run(self):
        try:
            if self.method == "k-anonymity":
                result, metrics = self.anonymizer.k_anonymize(
                    self.params['k'], 
                    self.params['suppression_threshold']
                )
            elif self.method == "t-closeness":
                result, metrics = self.anonymizer.t_closeness(
                    self.params['t'],
                    self.params['k'],
                    self.params['suppression_threshold']
                )
            elif self.method == "Differential Privacy":
                result, metrics = self.anonymizer.differential_privacy(
                    self.params['epsilon'],
                    self.params['sensitivity']
                )
            
            self.finished.emit(result, metrics)
        except Exception as e:
            self.finished.emit(None, str(e))

class DataAnonymizer:
    def __init__(self, data):
        self.original_data = data
        self.anonymized_data = None
        self.quasi_identifiers = ['Age', 'Workclass', 'Education', 'Marital Status', 'Occupation', 
                                 'Relationship', 'Race', 'Sex', 'Hours per week', 'Country']
        self.sensitive_attributes = ['Target', 'Capital Gain', 'Capital Loss']
        self.explicit_identifiers = ['Name', 'DOB', 'SSN', 'Zip']
    
    def k_anonymize(self, k, suppression_threshold=0.1):
        """Implement k-anonymity using generalization and suppression"""
        data = self.original_data.copy()
        
        # Remove explicit identifiers
        data = data.drop(columns=self.explicit_identifiers, errors='ignore')
        
        # Generalize quasi-identifiers
        data['Age'] = data['Age'].apply(lambda x: f"{int(x/10)*10}-{int(x/10)*10+9}")
        data['Hours per week'] = data['Hours per week'].apply(lambda x: f"{int(x/10)*10}-{int(x/10)*10+9}")
        
        # Find equivalence classes
        groups = data.groupby(self.quasi_identifiers)
        
        # Apply suppression to small groups
        suppressed_indices = []
        for name, group in groups:
            if len(group) < k:
                suppressed_indices.extend(group.index)
        
        # Apply suppression if needed
        if len(suppressed_indices) / len(data) > suppression_threshold:
            return None, "Suppression threshold exceeded. Try lower k or adjust suppression threshold."
        
        self.anonymized_data = data.drop(suppressed_indices)
        
        # Calculate metrics
        group_sizes = [len(group) for _, group in groups]
        min_group_size = min(group_sizes) if group_sizes else 0
        avg_group_size = np.mean(group_sizes) if group_sizes else 0
        suppression_rate = len(suppressed_indices) / len(data)
        
        metrics = {
            'k_achieved': min_group_size if min_group_size >= k else min_group_size,
            'avg_group_size': avg_group_size,
            'suppression_rate': suppression_rate,
            'num_groups': len(groups),
            'info_loss': self.calculate_info_loss(data, self.anonymized_data),
            'privacy_risk': {
                'reidentification_risk': 1/avg_group_size,
                'attribute_disclosure': self.calculate_attribute_disclosure()
            },
            'data_utility': {
                'column_preservation': self.calculate_column_preservation(data),
                'value_distortion': self.calculate_value_distortion(data)
            }
        }
        
        return self.anonymized_data, metrics

    def calculate_attribute_disclosure(self):
        """Calculate risk of sensitive attribute disclosure"""
        if not hasattr(self, 'anonymized_data') or self.anonymized_data is None:
            return 1.0  # Maximum risk if no anonymization
        
        if 'Target' not in self.anonymized_data.columns:
            return 0.0  # No sensitive attribute to disclose
        
        unique_values = self.anonymized_data['Target'].nunique()
        total_values = len(self.anonymized_data)
        return 1 - (unique_values / max(1, total_values))

    def calculate_column_preservation(self, original_data):
        """Calculate fraction of columns preserved"""
        if not hasattr(self, 'anonymized_data'):
            return 0.0
        return len(self.anonymized_data.columns) / len(original_data.columns)

    def calculate_value_distortion(self, original_data):
        """Memory-efficient calculation of value distortion"""
        if not hasattr(self, 'anonymized_data') or self.anonymized_data is None:
            return 1.0  # Maximum distortion
        
        numerical_cols = ['Age', 'Hours per week', 'Education-Num']
        total_diff = 0.0
        count = 0
        
        for col in numerical_cols:
            if col in original_data.columns and col in self.anonymized_data.columns:
                try:
                    # Process original values
                    if original_data[col].dtype == 'object' and '-' in str(original_data[col].iloc[0]):
                        orig_values = original_data[col].str.split('-').str[0].astype(float)
                    else:
                        orig_values = original_data[col].astype(float)
                    
                    # Process anonymized values
                    if self.anonymized_data[col].dtype == 'object' and '-' in str(self.anonymized_data[col].iloc[0]):
                        anon_values = self.anonymized_data[col].str.split('-').str[0].astype(float)
                    else:
                        anon_values = self.anonymized_data[col].astype(float)
                    
                    # Calculate mean absolute percentage difference
                    mean_orig = orig_values.mean()
                    if mean_orig == 0:
                        continue  # Skip to avoid division by zero
                    
                    abs_diff = (orig_values - anon_values).abs()
                    diff = abs_diff.mean() / mean_orig
                    total_diff += diff
                    count += 1
                    
                except (ValueError, TypeError):
                    continue  # Skip if conversion fails
        
        return total_diff / count if count > 0 else 0.0

    def t_closeness(self, t, k=5, suppression_threshold=0.1):
        """Complete robust t-closeness implementation that handles all edge cases"""
        # 1. Prepare data with explicit type handling
        data = self.original_data.copy()
        
        # Convert numeric columns safely
        numeric_cols = ['Age', 'Hours per week', 'Education-Num', 
                    'Capital Gain', 'Capital Loss']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # 2. Apply k-anonymity with bulletproof generalization
        def generalize_col(col_series):
            generalized = []
            for val in col_series:
                try:
                    num = float(val)
                    lower = int(num // 10 * 10)
                    generalized.append(f"{lower}-{lower+9}")
                except (ValueError, TypeError):
                    # If already generalized or non-numeric, keep as-is
                    if isinstance(val, str) and val.count('-') == 1:
                        generalized.append(val)
                    else:
                        generalized.append(str(val))
            return generalized
        
        if 'Age' in data.columns:
            data['Age'] = generalize_col(data['Age'])
        if 'Hours per week' in data.columns:
            data['Hours per week'] = generalize_col(data['Hours per week'])
        
        # 3. Group and suppress (k-anonymity phase)
        groups = data.groupby(self.quasi_identifiers)
        suppressed_indices = []
        
        for name, group in groups:
            if len(group) < k:
                suppressed_indices.extend(group.index)
        
        if len(suppressed_indices) / len(data) > suppression_threshold:
            return None, f"Suppression would remove {len(suppressed_indices)/len(data):.1%} (threshold {suppression_threshold:.1%})"
        
        anonymized_data = data.drop(suppressed_indices)
        
        # 4. t-closeness calculation with complete safety
        problematic_groups = []
        overall_dists = {
            attr: self.original_data[attr].value_counts(normalize=True)
            for attr in self.sensitive_attributes
            if attr in self.original_data.columns
        }
        
        for group_name, group in anonymized_data.groupby(self.quasi_identifiers):
            for attr, overall_dist in overall_dists.items():
                if attr not in group.columns:
                    continue
                    
                try:
                    # Calculate EMD safely
                    group_dist = group[attr].value_counts(normalize=True)
                    all_cats = sorted(set(overall_dist.index).union(set(group_dist.index)))
                    
                    emd = 0.5 * sum(
                        abs(overall_dist.get(cat, 0) - group_dist.get(cat, 0))
                        for cat in all_cats
                    )
                    
                    if emd > t:
                        problematic_groups.append((group_name, attr, emd))
                        break
                except Exception as e:
                    print(f"⚠️ Error processing group {group_name} for attribute {attr}:")
                    print(f"Problematic group contents:")
                    print(group[[attr]].to_string())
                    print(f"Original records in group:")
                    for idx in group.index:
                        print(f"\n--- Record {idx} ---")
                        print(self.original_data.loc[idx].to_string())
                    raise RuntimeError(f"Failed to process attribute {attr}") from e
        
        # 5. Prepare metrics
        metrics = {
            't_achieved': max((x[2] for x in problematic_groups), default=0),
            'violation_rate': len(problematic_groups)/max(1, len(groups)),
            'num_problematic_groups': len(problematic_groups),
            'suppressed_records': len(suppressed_indices),
            'effective_k': k,
            'guaranteed_t': t
        }
        
        return anonymized_data, metrics

    def differential_privacy(self, epsilon, sensitivity=1):
        """Implement differential privacy using Laplace mechanism"""
        data = self.original_data.copy()
        
        # Remove explicit identifiers
        data = data.drop(columns=self.explicit_identifiers, errors='ignore')
        
        # Apply noise to numerical columns
        numerical_cols = ['Age', 'Education-Num', 'Hours per week', 'Capital Gain', 'Capital Loss']
        for col in numerical_cols:
            if col in data.columns:
                scale = sensitivity / epsilon
                noise = np.random.laplace(0, scale, len(data))
                data[col] = data[col] + noise
        
        # Calculate utility metrics
        utility_metrics = self.calculate_utility_metrics(self.original_data, data)
        
        self.anonymized_data = data
        
        metrics = {
            'epsilon_used': epsilon,
            'sensitivity': sensitivity,
            'utility_metrics': utility_metrics,
            'privacy_risk': self.calculate_privacy_risk(self.original_data, data, None, epsilon)
        }
        
        return self.anonymized_data, metrics
    
    def calculate_info_loss(self, original, anonymized):
        """Calculate information loss metric"""
        loss = 0
        num_records = len(anonymized)
        
        if num_records == 0:
            return float('inf')
        
        # For each quasi-identifier, calculate the generalization level
        for col in self.quasi_identifiers:
            if col in original.columns and col in anonymized.columns:
                if original[col].dtype == 'object' and anonymized[col].dtype == 'object':
                    # For categorical data, check if values have been generalized
                    orig_values = original[col].value_counts(normalize=True)
                    anon_values = anonymized[col].value_counts(normalize=True)
                    
                    # Calculate KL divergence
                    all_categories = set(orig_values.index).union(set(anon_values.index))
                    for cat in all_categories:
                        if cat not in orig_values:
                            orig_values[cat] = 0
                        if cat not in anon_values:
                            anon_values[cat] = 0
                    
                    orig_values = orig_values.sort_index()
                    anon_values = anon_values.sort_index()
                    
                    # Avoid division by zero
                    kl_div = stats.entropy(orig_values + 1e-10, anon_values + 1e-10)
                    loss += kl_div
                else:
                    # For numerical data, calculate normalized absolute difference
                    orig_mean = original[col].mean()
                    anon_mean = anonymized[col].mean()
                    orig_std = original[col].std()
                    
                    if orig_std > 0:
                        loss += abs(orig_mean - anon_mean) / orig_std
        
        return loss / len(self.quasi_identifiers)
    
    def calculate_privacy_risk(self, original, anonymized, k=None, epsilon=None):
        """Calculate privacy risk metrics"""
        risk_metrics = {}
        
        if k is not None:
            # For k-anonymity/t-closeness
            if len(anonymized) == 0:
                return {'reidentification_risk': 1.0, 'attribute_disclosure': 1.0}
            
            # Re-identification risk (1/avg_group_size)
            groups = anonymized.groupby(self.quasi_identifiers)
            avg_group_size = np.mean([len(group) for _, group in groups]) if len(groups) > 0 else 1
            risk_metrics['reidentification_risk'] = 1 / avg_group_size
            
            # Attribute disclosure risk
            sensitive_col = self.sensitive_attributes[0] if self.sensitive_attributes else None
            if sensitive_col:
                diversity = anonymized[sensitive_col].nunique() / anonymized[sensitive_col].count()
                risk_metrics['attribute_disclosure'] = 1 - diversity
        
        if epsilon is not None:
            # For differential privacy
            risk_metrics['privacy_budget_used'] = epsilon
            risk_metrics['privacy_guarantee'] = f"ε={epsilon}"
        
        return risk_metrics
    
    def calculate_utility_metrics(self, original, anonymized):
        """Calculate various utility metrics including classification accuracy"""
        utility_metrics = {}
        
        # Basic statistical utility
        numerical_cols = ['Age', 'Education-Num', 'Hours per week', 'Capital Gain', 'Capital Loss']
        for col in numerical_cols:
            if col in original.columns and col in anonymized.columns:
                orig_mean = original[col].mean()
                anon_mean = anonymized[col].mean()
                orig_std = original[col].std()
                
                utility_metrics[f'{col}_mean_diff'] = abs(orig_mean - anon_mean)
                utility_metrics[f'{col}_std_diff'] = abs(orig_std - anonymized[col].std())
                if orig_std > 0:
                    utility_metrics[f'{col}_normalized_diff'] = abs(orig_mean - anon_mean) / orig_std
        
        # Classification accuracy (if target variable exists)
        if 'Target' in original.columns and 'Target' in anonymized.columns and len(anonymized) > 0:
            try:
                # Prepare data
                X_orig = original[self.quasi_identifiers].copy()
                y_orig = original['Target']
                
                X_anon = anonymized[self.quasi_identifiers].copy()
                y_anon = anonymized['Target']
                
                # Convert categorical to numerical
                for col in X_orig.select_dtypes(include=['object']).columns:
                    X_orig[col] = X_orig[col].astype('category').cat.codes
                    X_anon[col] = X_anon[col].astype('category').cat.codes
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_orig, y_orig, test_size=0.3, random_state=42
                )
                
                # Train model
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate on original test data
                orig_pred = model.predict(X_test)
                orig_accuracy = accuracy_score(y_test, orig_pred)
                
                # Evaluate on anonymized data
                anon_pred = model.predict(X_anon)
                anon_accuracy = accuracy_score(y_anon, anon_pred)
                
                utility_metrics['classification_accuracy_original'] = orig_accuracy
                utility_metrics['classification_accuracy_anonymized'] = anon_accuracy
                utility_metrics['classification_accuracy_diff'] = abs(orig_accuracy - anon_accuracy)
                
            except Exception as e:
                utility_metrics['classification_error'] = str(e)
        
        return utility_metrics

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Data Anonymization Tool")
        self.setGeometry(100, 100, 1200, 900)
        
        self.data = None
        self.anonymizer = None
        self.current_method = None
        self.anonymized_result = None
        self.thread = None
        
        self.initUI()
        self.setup_tooltips()
    
    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # File selection
        file_group = QGroupBox("Data Selection")
        file_layout = QHBoxLayout()
        
        self.file_label = QLabel("No file selected")
        btn_select = QPushButton("Select CSV File")
        btn_select.clicked.connect(self.select_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(btn_select)
        file_group.setLayout(file_layout)
        
        # Method selection
        method_group = QGroupBox("Anonymization Method")
        method_layout = QVBoxLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["k-anonymity", "t-closeness", "Differential Privacy"])
        self.method_combo.currentTextChanged.connect(self.update_parameters)
        
        method_layout.addWidget(self.method_combo)
        
        # Parameters (will be updated based on method)
        self.param_group = QGroupBox("Parameters")
        self.param_layout = QVBoxLayout()
        self.param_group.setLayout(self.param_layout)
        
        method_layout.addWidget(self.param_group)
        method_group.setLayout(method_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        
        # Action buttons
        btn_group = QGroupBox()
        btn_layout = QHBoxLayout()
        
        btn_anonymize = QPushButton("Anonymize")
        btn_anonymize.clicked.connect(self.anonymize_data)
        
        btn_export = QPushButton("Export Result")
        btn_export.clicked.connect(self.export_result)
        
        btn_layout.addWidget(btn_anonymize)
        btn_layout.addWidget(btn_export)
        btn_group.setLayout(btn_layout)
        
        # Results display
        self.result_tabs = QTabWidget()
        
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.metrics_table = QTextEdit()
        self.metrics_table.setReadOnly(True)
        
        self.result_tabs.addTab(self.text_output, "Text Report")
        self.result_tabs.addTab(self.canvas, "Visualization")
        self.result_tabs.addTab(self.metrics_table, "Detailed Metrics")
        
        # Assemble main layout
        main_layout.addWidget(file_group)
        main_layout.addWidget(method_group)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(btn_group)
        main_layout.addWidget(self.result_tabs)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initialize parameters
        self.update_parameters()
    
    def setup_tooltips(self):
        # Method selection tooltip
        self.method_combo.setToolTip(
            "Select anonymization method:\n"
            "- k-anonymity: Ensures each record is indistinguishable from at least k-1 others\n"
            "- t-closeness: Ensures distribution of sensitive attributes within each group is close to overall distribution\n"
            "- Differential Privacy: Adds noise to data to provide strong mathematical privacy guarantees"
        )
    
    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", 
                                                 "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.file_label.setText(file_name)
            try:
                self.data = pd.read_csv(file_name)
                self.anonymizer = DataAnonymizer(self.data)
                self.text_output.append("File loaded successfully!")
                self.text_output.append(f"Records: {len(self.data)}")
            except Exception as e:
                self.text_output.append(f"Error loading file: {str(e)}")
    
    def update_parameters(self):
        # Clear existing parameters
        for i in reversed(range(self.param_layout.count())): 
            widget = self.param_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        method = self.method_combo.currentText()
        self.current_method = method
        
        if method == "k-anonymity":
            lbl_k = QLabel("k value:")
            self.spin_k = QSpinBox()
            self.spin_k.setMinimum(2)
            self.spin_k.setMaximum(100)
            self.spin_k.setValue(5)
            self.spin_k.setToolTip("Minimum size of equivalence groups. Higher values provide more privacy but may reduce data utility.")
            
            lbl_supp = QLabel("Suppression threshold:")
            self.spin_supp = QDoubleSpinBox()
            self.spin_supp.setMinimum(0.01)
            self.spin_supp.setMaximum(0.5)
            self.spin_supp.setSingleStep(0.01)
            self.spin_supp.setValue(0.1)
            self.spin_supp.setToolTip("Maximum fraction of records that can be suppressed when creating equivalence classes.")
            
            self.param_layout.addWidget(lbl_k)
            self.param_layout.addWidget(self.spin_k)
            self.param_layout.addWidget(lbl_supp)
            self.param_layout.addWidget(self.spin_supp)
        
        elif method == "t-closeness":
            lbl_t = QLabel("t value:")
            self.spin_t = QDoubleSpinBox()
            self.spin_t.setMinimum(0.01)
            self.spin_t.setMaximum(1.0)
            self.spin_t.setSingleStep(0.01)
            self.spin_t.setValue(0.2)
            self.spin_t.setToolTip("Maximum allowed Earth Mover's Distance between group and overall distributions of sensitive attributes.")
            
            lbl_k = QLabel("Initial k value:")
            self.spin_k = QSpinBox()
            self.spin_k.setMinimum(2)
            self.spin_k.setMaximum(100)
            self.spin_k.setValue(5)
            self.spin_k.setToolTip("Minimum size of equivalence groups before applying t-closeness.")
            
            lbl_supp = QLabel("Suppression threshold:")
            self.spin_supp = QDoubleSpinBox()
            self.spin_supp.setMinimum(0.01)
            self.spin_supp.setMaximum(0.5)
            self.spin_supp.setSingleStep(0.01)
            self.spin_supp.setValue(0.1)
            self.spin_supp.setToolTip("Maximum fraction of records that can be suppressed when creating equivalence classes.")
            
            self.param_layout.addWidget(lbl_t)
            self.param_layout.addWidget(self.spin_t)
            self.param_layout.addWidget(lbl_k)
            self.param_layout.addWidget(self.spin_k)
            self.param_layout.addWidget(lbl_supp)
            self.param_layout.addWidget(self.spin_supp)
        
        elif method == "Differential Privacy":
            lbl_eps = QLabel("Epsilon (ε):")
            self.spin_eps = QDoubleSpinBox()
            self.spin_eps.setMinimum(0.01)
            self.spin_eps.setMaximum(10.0)
            self.spin_eps.setSingleStep(0.1)
            self.spin_eps.setValue(1.0)
            self.spin_eps.setToolTip("Privacy budget. Lower values provide stronger privacy but add more noise.")
            
            lbl_sens = QLabel("Sensitivity:")
            self.spin_sens = QDoubleSpinBox()
            self.spin_sens.setMinimum(0.1)
            self.spin_sens.setMaximum(100.0)
            self.spin_sens.setSingleStep(0.1)
            self.spin_sens.setValue(1.0)
            self.spin_sens.setToolTip("Maximum influence a single record can have on the output.")
            
            self.param_layout.addWidget(lbl_eps)
            self.param_layout.addWidget(self.spin_eps)
            self.param_layout.addWidget(lbl_sens)
            self.param_layout.addWidget(self.spin_sens)
    
    def anonymize_data(self):
        if self.data is None:
            self.text_output.append("Please select a file first!")
            return
        
        method = self.current_method
        self.text_output.append(f"\nApplying {method}...")
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        
        # Prepare parameters based on method
        params = {}
        if method == "k-anonymity":
            params = {
                'k': self.spin_k.value(),
                'suppression_threshold': self.spin_supp.value()
            }
        elif method == "t-closeness":
            params = {
                't': self.spin_t.value(),
                'k': self.spin_k.value(),
                'suppression_threshold': self.spin_supp.value()
            }
        elif method == "Differential Privacy":
            params = {
                'epsilon': self.spin_eps.value(),
                'sensitivity': self.spin_sens.value()
            }
        
        # Create and start worker thread
        self.thread = AnonymizationThread(self.anonymizer, method, params)
        self.thread.finished.connect(self.handle_anonymization_result)
        self.thread.start()
        
        # Simulate progress updates
        self.simulate_progress()
    
    def simulate_progress(self):
        """Simulate progress updates for the anonymization process"""
        for i in range(1, 101):
            time.sleep(0.05)  # Simulate work being done
            self.progress_bar.setValue(i)
            QApplication.processEvents()
    
    def handle_anonymization_result(self, result, metrics_or_error):
        self.progress_bar.setVisible(False)
        
        if result is None:
            self.text_output.append(f"Anonymization failed: {metrics_or_error}")
            return
        
        metrics = metrics_or_error
        self.anonymized_result = result
        
        # Display basic results
        self.text_output.append(f"\n{self.current_method} applied successfully!")
        self.text_output.append(f"Records remaining: {len(result)}/{len(self.data)} ({len(result)/len(self.data):.1%})")
        
        # Update metrics display
        self.update_metrics_table(metrics)
        
        # Visualizations
        self.update_visualizations(result)

    def update_metrics_table(self, metrics):
        """Helper method to update metrics display"""
        metrics_text = "=== Privacy Metrics ===\n"
        for key, value in metrics.items():
            if key != 'utility_metrics':
                if isinstance(value, dict):
                    metrics_text += f"\n{key}:\n"
                    for subkey, subvalue in value.items():
                        try:
                            metrics_text += f"  {subkey}: {float(subvalue):.4f}\n"
                        except (ValueError, TypeError):
                            metrics_text += f"  {subkey}: {subvalue}\n"
                else:
                    try:
                        metrics_text += f"{key}: {float(value):.4f}\n"
                    except (ValueError, TypeError):
                        metrics_text += f"{key}: {value}\n"
        
        if 'utility_metrics' in metrics:
            metrics_text += "\n=== Utility Metrics ===\n"
            for key, value in metrics['utility_metrics'].items():
                try:
                    metrics_text += f"{key}: {float(value):.4f}\n"
                except (ValueError, TypeError):
                    metrics_text += f"{key}: {value}\n"
        
        self.metrics_table.setPlainText(metrics_text)
    
    def update_visualizations(self, result):
        """Handle all visualization updates"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        try:
            if self.current_method == "k-anonymity":
                # Visualization code for k-anonymity
                groups = result.groupby(self.anonymizer.quasi_identifiers)
                group_sizes = [len(group) for _, group in groups]
                
                ax.hist(group_sizes, bins=range(min(group_sizes), max(group_sizes)+1), 
                    align='left', rwidth=0.8)
                ax.axvline(x=self.spin_k.value(), color='r', linestyle='--', 
                        label=f'k={self.spin_k.value()}')
                ax.set_xlabel('Equivalence Class Size')
                ax.set_ylabel('Frequency')
                ax.legend()

            elif self.current_method == "t-closeness":
                # Visualization code for t-closeness
                emd_values = []
                groups = result.groupby(self.anonymizer.quasi_identifiers)
                
                for _, group in groups:
                    for sensitive_attr in self.anonymizer.sensitive_attributes:
                        if sensitive_attr in group.columns:
                            overall_dist = (self.anonymizer.original_data[sensitive_attr]
                                        .value_counts(normalize=True))
                            group_dist = group[sensitive_attr].value_counts(normalize=True)
                            
                            all_cats = set(overall_dist.index).union(set(group_dist.index))
                            emd = 0.5 * sum(abs(overall_dist.reindex(all_cats, fill_value=0) - 
                                            group_dist.reindex(all_cats, fill_value=0)))
                            emd_values.append(emd)
                
                if emd_values:
                    ax.hist(emd_values, bins=20, alpha=0.7)
                    ax.axvline(x=self.spin_t.value(), color='r', linestyle='--', 
                            label=f't={self.spin_t.value():.2f}')
                    ax.set_xlabel('Earth Mover\'s Distance (EMD)')
                    ax.set_ylabel('Frequency')
                    ax.legend()

            elif self.current_method == "Differential Privacy":
                # Visualization code for differential privacy
                numerical_cols = ['Age', 'Hours per week', 'Education-Num', 
                                'Capital Gain', 'Capital Loss']
                col_to_plot = next((col for col in numerical_cols 
                                if col in self.anonymizer.original_data.columns), None)
                
                if col_to_plot:
                    orig = self.anonymizer.original_data[col_to_plot]
                    anon = result[col_to_plot]
                    
                    ax.hist(orig, bins=20, alpha=0.5, label='Original')
                    ax.hist(anon, bins=20, alpha=0.5, label='Anonymized')
                    ax.set_xlabel(col_to_plot)
                    ax.set_ylabel('Frequency')
                    ax.legend()
            
            self.canvas.draw()
            
        except Exception as e:
            self.text_output.append(f"Visualization error: {str(e)}")

    def export_result(self):
        if not hasattr(self, 'anonymized_result') or self.anonymized_result is None:
            self.text_output.append("No anonymized data to export!")
            return
        
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Anonymized Data", "", 
                                                  "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            try:
                self.anonymized_result.to_csv(file_name, index=False)
                self.text_output.append(f"Anonymized data saved to {file_name}")
            except Exception as e:
                self.text_output.append(f"Error saving file: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()