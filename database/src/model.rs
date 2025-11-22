use std::{collections::HashMap};
use ort::{inputs, session::{InMemorySession, Session, SessionOutputs, builder::GraphOptimizationLevel}, value::Value};
use crate::{cache::{Evictor, Entry}, common::{log1p, unix_epoch, Result}};

pub struct DeepQNetwork<'a> {
	model: InMemorySession<'a>
}

impl<'a> DeepQNetwork<'a> {
	pub fn new() -> Result<Self> {
		Ok(DeepQNetwork {
			model: Session::builder()?
			.with_optimization_level(GraphOptimizationLevel::Level3)?
			.commit_from_memory_directly(include_bytes!("../model.onnx"))?
		})
	}
}

impl<'a> Evictor for DeepQNetwork<'a> {
	fn select_victim(self: &mut Self, entries: &HashMap<String, Entry>) -> Result<String> {
		let length: usize = entries.len();

		if length == 0 {
			return Err(Box::from("entries length must be greater than zero"));
		}

		let mut keys: Vec<&String> = Vec::with_capacity(length);
		let mut inputs: Vec<f32> = Vec::with_capacity(length * 4);
		let capacity: f32 = log1p(entries.capacity() as u64);

		for entry in entries {
			keys.push(entry.0);
			inputs.push(log1p(unix_epoch()? - entry.1.accessed_at));
			inputs.push(log1p(entry.1.access_count));
			inputs.push(log1p(entry.1.value.len() as u64));
			inputs.push(capacity);
		}

		let output: SessionOutputs = self.model.run(inputs!["args_0" => Value::from_array((([length, 4]), inputs))?])?;

		let mut i: usize = 0;
		let mut minimum_score: f32 = f32::MAX;
		let mut minimum_index: usize = 0;

		for score in output[0].try_extract_tensor::<f32>()?.1 {
			if *score < minimum_score {
				minimum_score = *score;
				minimum_index = i;
			}

			i += 1;
		}

		Ok(keys[minimum_index].clone())
	}
}

pub struct LeastRecentlyUsed {}

impl Evictor for LeastRecentlyUsed {
	fn select_victim(self: &mut Self, entries: &HashMap<String, Entry>) -> Result<String> {
		if entries.len() == 0 {
			return Err(Box::from("entries length must be greater than zero"));
		}

		let mut minimum_accessed_at: u64 = u64::MAX;
		let mut minimum_key: &String = &String::new();

		for entry in entries {
			if entry.1.accessed_at < minimum_accessed_at {
				minimum_accessed_at = entry.1.accessed_at;
				minimum_key = entry.0;
			} 
		}

		Ok(minimum_key.clone())
	}
}

pub struct LeastFrequentlyUsed {}

impl Evictor for LeastFrequentlyUsed {
	fn select_victim(self: &mut Self, entries: &HashMap<String, Entry>) -> Result<String> {
		if entries.len() == 0 {
			return Err(Box::from("entries length must be greater than zero"));
		}

		let mut minimum_access_count: u64 = u64::MAX;
		let mut minimum_key: &String = &String::new();

		for entry in entries {
			if entry.1.access_count < minimum_access_count {
				minimum_access_count = entry.1.access_count;
				minimum_key = entry.0;
			} 
		}

		Ok(minimum_key.clone())
	}
}